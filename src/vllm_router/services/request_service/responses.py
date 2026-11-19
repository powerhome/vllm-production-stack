"""Resolve vLLM Responses API resources to their engine-local store."""

import asyncio
import json
import re
from collections.abc import Iterable, Mapping
from http import HTTPStatus
from urllib.parse import quote

import aiohttp
from fastapi import HTTPException

from vllm_router.service_discovery import EndpointInfo

_RESPONSES_PATH = "/v1/responses/"
_RESPONSES_RESOURCE = re.compile(r"/v1/responses/([^/]+)(?:/cancel)?")
_RESPONSE_ID = re.compile(r"[^/\\\x00-\x1f\x7f]+")
_OWNER_LOOKUP_WORKERS = 8
_OWNER_LOOKUP_TIMEOUT = 5.0
_OWNER_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=2.0)
_MAX_PROBE_BODY = 8 * 1024 * 1024


def _valid_response_id(response_id: str) -> bool:
    return (
        isinstance(response_id, str)
        and response_id not in {".", ".."}
        and _RESPONSE_ID.fullmatch(response_id) is not None
    )


def response_resource_id(path: str) -> str | None:
    """Extract an ID from a decoded Responses API retrieval or cancellation path."""
    match = _RESPONSES_RESOURCE.fullmatch(path)
    if match and _valid_response_id(match[1]):
        return match[1]
    return None


def response_resource_path(response_id: str) -> str:
    """Return the canonical retrieval path for an opaque response ID."""
    if _valid_response_id(response_id):
        try:
            return _RESPONSES_PATH + quote(response_id, safe="")
        except UnicodeEncodeError:
            pass
    raise HTTPException(HTTPStatus.BAD_REQUEST, "Invalid response ID.")


async def _probe(
    client: aiohttp.ClientSession,
    url: str,
    response_id: str,
    headers: Mapping[str, str],
) -> int:
    try:
        async with client.get(
            url,
            headers=headers,
            params={"stream": "false"},
            allow_redirects=False,
            timeout=_OWNER_PROBE_TIMEOUT,
            raise_for_status=False,
        ) as response:
            if response.status != HTTPStatus.OK:
                return response.status
            if response.content_type != "application/json":
                return HTTPStatus.SERVICE_UNAVAILABLE

            body = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                if len(body) + len(chunk) > _MAX_PROBE_BODY:
                    return HTTPStatus.SERVICE_UNAVAILABLE
                body.extend(chunk)
            item = json.loads(body)
            if (
                isinstance(item, dict)
                and item.get("id") == response_id
                and item.get("object") == "response"
            ):
                return HTTPStatus.OK
    except (aiohttp.ClientError, TimeoutError, ValueError, RecursionError):
        pass
    return HTTPStatus.SERVICE_UNAVAILABLE


async def find_response_owner(
    client: aiohttp.ClientSession,
    endpoints: Iterable[EndpointInfo],
    response_id: str,
    headers: Mapping[str, str],
) -> EndpointInfo:
    """Return the sole verified owner after probing the supplied candidate engines."""
    path = response_resource_path(response_id)
    targets: dict[str, EndpointInfo] = {}
    for endpoint in endpoints:
        targets.setdefault(endpoint.url.rstrip("/") + path, endpoint)

    # Each probe is a fresh read. In particular, the original request may be a
    # POST or a streaming GET, or may carry conditional or range headers.
    probe_headers = {
        name: value
        for name, value in headers.items()
        if name.lower() in {"authorization", "cookie", "x-api-key"}
    }
    probe_headers["Accept"] = "application/json"

    pending = iter(targets.items())
    owners = []
    statuses: set[int] = set()

    async def probe_targets() -> None:
        for url, endpoint in pending:
            status = await _probe(client, url, response_id, probe_headers)
            statuses.add(status)
            if status == HTTPStatus.OK:
                owners.append(endpoint)

    try:
        async with asyncio.timeout(_OWNER_LOOKUP_TIMEOUT):
            async with asyncio.TaskGroup() as group:
                for _ in range(min(_OWNER_LOOKUP_WORKERS, len(targets))):
                    group.create_task(probe_targets())
    except TimeoutError:
        statuses.add(HTTPStatus.SERVICE_UNAVAILABLE)

    if len(owners) > 1:
        raise HTTPException(
            HTTPStatus.CONFLICT, "Multiple engines own this response ID."
        )
    auth_failure = statuses & {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
    unexpected = statuses - {HTTPStatus.OK, HTTPStatus.NOT_FOUND} - auth_failure
    if not targets or unexpected or (owners and auth_failure):
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE, "Response owner lookup did not complete."
        )

    if owners:
        return owners[0]
    if auth_failure:
        raise HTTPException(min(auth_failure), "Response owner lookup was denied.")
    raise HTTPException(HTTPStatus.NOT_FOUND, "Response ID not found.")
