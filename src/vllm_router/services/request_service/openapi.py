"""Discover the HTTP inference operations advertised by vLLM engines."""

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Any

import aiohttp

from vllm_router.service_discovery import EndpointInfo

_JSON = "application/json"
_MULTIPART = "multipart/form-data"
_SCHEMA_CACHE_SIZE = 256
_SCHEMA_CACHE_TTL = 60
_SCHEMA_FAILURE_TTL = 10
_SCHEMA_SIZE_LIMIT = 4 * 1024 * 1024
_SCHEMA_TIMEOUT = 2
_DISCOVERY_TIMEOUT = 5
_DISCOVERY_CONCURRENCY = 8
_SCHEMA_DEPTH_LIMIT = 12

_RESPONSES_READ = ("GET", "/v1/responses/{response_id}")
_RESPONSES_CANCEL = ("POST", "/v1/responses/{response_id}/cancel")
_RESPONSES_OPERATION_IDS = {
    _RESPONSES_READ: "retrieve_responses_v1_responses__response_id__get",
    _RESPONSES_CANCEL: "cancel_responses_v1_responses__response_id__cancel_post",
}


@dataclass(frozen=True)
class InferenceOperation:
    model_optional: bool


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _array(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dereference(document: dict, value: Any) -> dict:
    for _ in range(_SCHEMA_DEPTH_LIMIT):
        value = _object(value)
        if "$ref" not in value:
            return value
        ref = value["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            break
        value = document
        for segment in ref[2:].split("/"):
            value = _object(value).get(segment.replace("~1", "/").replace("~0", "~"))
    return {}


def _model_type(document: dict, schema: Any, memo: dict, depth: int = 0) -> bool:
    if depth >= _SCHEMA_DEPTH_LIMIT:
        return False
    schema = _dereference(document, schema)
    if not schema:
        return False

    cache_key = (id(schema), depth, "type")
    if cache_key not in memo:
        field_type = schema.get("type")
        memo[cache_key] = (
            field_type == "string"
            or (isinstance(field_type, list) and "string" in field_type)
            or any(
                _model_type(document, part, memo, depth + 1)
                for key in ("anyOf", "oneOf", "allOf")
                for part in _array(schema.get(key))
            )
        )
    return memo[cache_key]


def _model_field(
    document: dict,
    schema: Any,
    memo: dict,
    depth: int = 0,
) -> tuple[bool, bool]:
    """Return whether a schema declares a string model field and requires it."""
    if depth >= _SCHEMA_DEPTH_LIMIT:
        return False, False
    schema = _dereference(document, schema)
    if not schema:
        return False, False

    cache_key = (id(schema), depth, "field")
    if cache_key in memo:
        return memo[cache_key]

    has_model = _model_type(
        document, _object(schema.get("properties")).get("model"), memo
    )
    required = "model" in _array(schema.get("required"))
    for key in ("allOf", "anyOf", "oneOf"):
        branches = _array(schema.get(key))
        if not branches:
            continue
        fields = [_model_field(document, part, memo, depth + 1) for part in branches]
        combine = any if key == "allOf" else all
        has_model |= combine(has_field for has_field, _ in fields)
        required |= combine(needs_field for _, needs_field in fields)

    memo[cache_key] = (has_model, required)
    return has_model, required


def _path_parts(path: str) -> tuple[str, ...]:
    # Accept decoded path segments only. Never admit a path that can be
    # reinterpreted as a different operation when constructing the proxy URL.
    if not path.startswith("/") or any(
        char in "?#%\\" or char.isspace() or not char.isprintable() for char in path
    ):
        return ()
    parts = tuple(path[1:].split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        return ()
    return parts


def _template_parameter(part: str) -> bool:
    return part.startswith("{") and part.endswith("}") and part[1:-1].isidentifier()


def _responses_operation(document: dict, operation: dict, verb_path: tuple) -> bool:
    if operation.get("operationId") != _RESPONSES_OPERATION_IDS[verb_path]:
        return False
    for parameter in _array(operation.get("parameters")):
        parameter = _dereference(document, parameter)
        if (
            parameter.get("name") == "response_id"
            and parameter.get("in") == "path"
            and parameter.get("required") is True
        ):
            return True
    return False


@dataclass
class _PathOperations:
    verbs: set[str]
    inference: dict[str, InferenceOperation]
    responses_read: bool
    responses_cancel: bool


@dataclass
class _InferenceSchema:
    paths: dict[tuple[str, ...], _PathOperations]
    templates: list[tuple[tuple[str | None, ...], _PathOperations]]

    def lookup(
        self, method: str, path: tuple[str, ...], media_type: str, kind: str
    ) -> InferenceOperation | None:
        operations = self.paths.get(path)
        for template, candidate in self.templates:
            if method not in candidate.verbs or len(template) != len(path):
                continue
            if not all(
                part is None or part == actual for part, actual in zip(template, path)
            ):
                continue
            if operations is not None and method in operations.verbs:
                # A template and a different declared operation both match.
                # OpenAPI does not communicate the engine's ASGI route precedence.
                return None
            operations = candidate

        if operations is None:
            return None
        if kind == "inference" and method == "POST":
            return operations.inference.get(media_type)
        if kind == "responses-read" and method == "GET" and operations.responses_read:
            return InferenceOperation(model_optional=True)
        if (
            kind == "responses-cancel"
            and method == "POST"
            and operations.responses_cancel
        ):
            return InferenceOperation(model_optional=True)
        return None


def _parse_schema(document: Any) -> _InferenceSchema | None:
    document = _object(document)
    version = document.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        return None
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return None

    schema = _InferenceSchema(paths={}, templates=[])
    memo = {}
    for path, item in paths.items():
        parts = _path_parts(path)
        if not parts:
            continue
        item = _dereference(document, item)
        post = _dereference(document, item.get("post"))
        get = _dereference(document, item.get("get"))
        path_operations = _PathOperations(
            verbs={verb for verb in ("GET", "POST") if verb.lower() in item},
            inference={},
            responses_read=path == _RESPONSES_READ[1]
            and _responses_operation(document, get, _RESPONSES_READ),
            responses_cancel=path == _RESPONSES_CANCEL[1]
            and _responses_operation(document, post, _RESPONSES_CANCEL),
        )
        body = _dereference(document, post.get("requestBody"))
        for media_type in (_JSON, _MULTIPART):
            media = _object(_object(body.get("content")).get(media_type))
            has_model, required = _model_field(document, media.get("schema"), memo)
            if has_model:
                path_operations.inference[media_type] = InferenceOperation(
                    model_optional=not required
                )

        # SageMaker validates its raw Request against task-dependent inference
        # models at runtime, so FastAPI omits a requestBody for this operation.
        if (
            path == "/invocations"
            and post.get("operationId") == "invocations_invocations_post"
        ):
            path_operations.inference.setdefault(
                _JSON, InferenceOperation(model_optional=True)
            )

        if any(_template_parameter(part) for part in parts):
            template = tuple(
                None if _template_parameter(part) else part for part in parts
            )
            schema.templates.append((template, path_operations))
        else:
            schema.paths[parts] = path_operations

    return schema


class OpenAPIInferenceDiscovery:
    """Resolve operations on configured engine URLs using a bounded schema cache."""

    def __init__(self) -> None:
        self._cache: OrderedDict[str, tuple[float, _InferenceSchema | None]] = (
            OrderedDict()
        )
        self._locks = [asyncio.Lock() for _ in range(_DISCOVERY_CONCURRENCY)]

    async def _fetch(
        self, client: aiohttp.ClientSession, engine_url: str
    ) -> _InferenceSchema | None:
        try:
            async with client.get(
                engine_url.rstrip("/") + "/openapi.json",
                allow_redirects=False,
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
                timeout=aiohttp.ClientTimeout(total=_SCHEMA_TIMEOUT),
            ) as response:
                if response.status != 200:
                    return None
                data = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    data.extend(chunk)
                    if len(data) > _SCHEMA_SIZE_LIMIT:
                        return None
                return _parse_schema(json.loads(data))
        except (aiohttp.ClientError, TimeoutError, ValueError, RecursionError):
            return None

    async def _engine_schema(
        self, client: aiohttp.ClientSession, url: str
    ) -> _InferenceSchema | None:
        # Stripes bound both the number of simultaneous backend requests and
        # the memory used to coalesce concurrent cache misses for a URL.
        lock = self._locks[hash(url) % len(self._locks)]
        async with lock:
            entry = self._cache.get(url)
            if entry is not None and monotonic() < entry[0]:
                self._cache.move_to_end(url)
                return entry[1]

            schema = await self._fetch(client, url)
            ttl = _SCHEMA_CACHE_TTL if schema is not None else _SCHEMA_FAILURE_TTL
            self._cache[url] = (monotonic() + ttl, schema)
            self._cache.move_to_end(url)
            if len(self._cache) > _SCHEMA_CACHE_SIZE:
                self._cache.popitem(last=False)
            return schema

    async def eligible_endpoints(
        self,
        client: aiohttp.ClientSession,
        endpoints: list[EndpointInfo],
        method: str,
        path: str,
        media_type: str | None,
        *,
        kind: str = "inference",
    ) -> list[tuple[EndpointInfo, InferenceOperation]]:
        parts = _path_parts(path)
        method = method.upper()
        media = (media_type or _JSON).split(";", 1)[0].strip().lower()
        if not parts or method not in {"GET", "POST"}:
            return []
        if kind == "inference" and (
            method != "POST" or media not in {_JSON, _MULTIPART}
        ):
            return []

        urls = list(dict.fromkeys(endpoint.url for endpoint in endpoints))
        operations = {}

        async def inspect(url: str) -> None:
            schema = await self._engine_schema(client, url)
            if schema is not None:
                operation = schema.lookup(method, parts, media, kind)
                if operation is not None:
                    operations[url] = operation

        try:
            async with asyncio.timeout(_DISCOVERY_TIMEOUT):
                for start in range(0, len(urls), _DISCOVERY_CONCURRENCY):
                    async with asyncio.TaskGroup() as group:
                        for url in urls[start : start + _DISCOVERY_CONCURRENCY]:
                            group.create_task(inspect(url))
        except TimeoutError:
            pass

        return [
            (endpoint, operations[endpoint.url])
            for endpoint in endpoints
            if endpoint.url in operations
        ]
