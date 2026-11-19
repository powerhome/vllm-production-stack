from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp
import pytest
from aiohttp import web
from fastapi import FastAPI, Request
from pydantic import BaseModel

from vllm_router.service_discovery import EndpointInfo

JSON = "application/json"


class _RequiredModel(BaseModel):
    model: str


class _OptionalModel(BaseModel):
    model: str | None = None


class _AdapterManagement(BaseModel):
    lora_name: str
    lora_path: str


def _upstream_schema():
    app = FastAPI()

    # These signatures reproduce the relevant contract from vLLM v0.32.0:
    # request models for inference, raw Request for SageMaker invocations,
    # and required response_id path parameters for Responses resources.
    @app.post("/inference/unknown-to-router")
    async def new_inference(body: _RequiredModel | _OptionalModel):
        pass

    @app.post("/v1/load_lora_adapter")
    async def load_lora_adapter(body: _AdapterManagement):
        pass

    @app.post("/collective_rpc")
    async def collective_rpc(request: Request):
        pass

    @app.post("/invocations")
    async def invocations(request: Request):
        pass

    @app.get("/v1/responses/{response_id}")
    async def retrieve_responses(response_id: str, stream: bool = False):
        pass

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_responses(response_id: str):
        pass

    return app.openapi()


@asynccontextmanager
async def _serve_schema(schema=None, *, status_code=200, redirect=None):
    calls = []

    async def openapi(request):
        calls.append(request.path)
        if redirect:
            raise web.HTTPFound(location=redirect)
        return web.json_response(schema, status=status_code)

    app = web.Application()
    app.router.add_get("/openapi.json", openapi)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        _, port = runner.addresses[0]
        url = f"http://127.0.0.1:{port}"
        yield EndpointInfo(url, ["model"], str(port), 0.0, "", False), calls
    finally:
        await runner.cleanup()


async def test_discover_inference_on_only_engines_advertising_it():
    from vllm_router.services.request_service.openapi import OpenAPIInferenceDiscovery

    schema = _upstream_schema()
    async with (
        _serve_schema({"openapi": "3.1.0", "paths": {}}) as (absent, absent_calls),
        _serve_schema(schema) as (present, present_calls),
        _serve_schema(status_code=404) as (disabled, disabled_calls),
        aiohttp.ClientSession() as client,
    ):
        discovery = OpenAPIInferenceDiscovery()
        endpoints = [absent, present, disabled]

        admitted = await discovery.eligible_endpoints(
            client, endpoints, "POST", "/inference/unknown-to-router", JSON
        )
        assert [endpoint for endpoint, _ in admitted] == [present]
        assert admitted[0][1].model_optional
        assert (
            await discovery.eligible_endpoints(
                client, endpoints, "POST", "/v1/load_lora_adapter", JSON
            )
            == []
        )
        assert (
            await discovery.eligible_endpoints(
                client, endpoints, "POST", "/collective_rpc", JSON
            )
            == []
        )
        assert await discovery.eligible_endpoints(
            client, endpoints, "POST", "/invocations", JSON
        ) == [(present, admitted[0][1])]
        assert absent_calls == present_calls == disabled_calls == ["/openapi.json"]


@pytest.mark.parametrize(
    "method,path,kind,expected",
    [
        ("GET", "/v1/responses/resp_123", "responses-read", True),
        ("POST", "/v1/responses/resp_123/cancel", "responses-cancel", True),
        ("GET", "/v1/responses/resp_123", "inference", False),
        ("GET", "/v1/responses/resp_123/cancel", "responses-cancel", False),
        ("GET", "/v1/responses/../resp_123", "responses-read", False),
        ("GET", "/v1/responses/%2e%2e", "responses-read", False),
    ],
)
async def test_responses_operations_require_the_resource_protocol(
    method, path, kind, expected
):
    from vllm_router.services.request_service.openapi import OpenAPIInferenceDiscovery

    async with (
        _serve_schema(_upstream_schema()) as (engine, _),
        aiohttp.ClientSession() as client,
    ):
        admitted = await OpenAPIInferenceDiscovery().eligible_endpoints(
            client, [engine], method, path, None, kind=kind
        )
        assert bool(admitted) is expected


async def test_multipart_model_schema_and_management_template_ambiguity():
    from vllm_router.services.request_service.openapi import OpenAPIInferenceDiscovery

    schema = {
        "openapi": "3.1.0",
        "paths": {
            "/task/{name}": {
                "post": {
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "allOf": [
                                        {"$ref": "#/components/schemas/ModelParam"},
                                        {"required": ["model"]},
                                    ]
                                }
                            }
                        }
                    }
                }
            },
            "/task/management": {"post": {}},
        },
        "components": {
            "schemas": {
                "ModelParam": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                }
            }
        },
    }
    async with (
        _serve_schema(schema) as (engine, _),
        aiohttp.ClientSession() as client,
    ):
        discovery = OpenAPIInferenceDiscovery()
        admitted = await discovery.eligible_endpoints(
            client,
            [engine],
            "POST",
            "/task/transcription",
            "multipart/form-data; boundary=boundary",
        )
        assert [endpoint for endpoint, _ in admitted] == [engine]
        assert not admitted[0][1].model_optional
        assert (
            await discovery.eligible_endpoints(
                client, [engine], "POST", "/task/management", "multipart/form-data"
            )
            == []
        )
        assert (
            await discovery.eligible_endpoints(
                client, [engine], "POST", "/task/transcription", JSON
            )
            == []
        )


async def test_schema_redirect_is_refused_and_cached_by_engine_url():
    from vllm_router.services.request_service.openapi import OpenAPIInferenceDiscovery

    async with (
        _serve_schema(_upstream_schema()) as (target, target_calls),
        _serve_schema(redirect=f"{target.url}/openapi.json") as (engine, engine_calls),
        aiohttp.ClientSession() as client,
    ):
        discovery = OpenAPIInferenceDiscovery()
        endpoints = [engine, replace(engine, model_names=["another-model"])]
        for _ in range(2):
            admitted = await discovery.eligible_endpoints(
                client, endpoints, "POST", "/inference/unknown-to-router", JSON
            )
            assert admitted == []
        assert engine_calls == ["/openapi.json"]
        assert target_calls == []


async def test_local_body_reference_and_cyclic_untyped_schema():
    from vllm_router.services.request_service.openapi import OpenAPIInferenceDiscovery

    schema = {
        "openapi": "3.1.0",
        "paths": {
            "/task": {"post": {"requestBody": {"$ref": "#/x-test/Request"}}},
            "/cyclic": {
                "post": {
                    "requestBody": {
                        "content": {
                            JSON: {
                                "schema": {
                                    "properties": {
                                        "model": {"$ref": "#/components/schemas/Cycle"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
        },
        "x-test": {
            "Request": {
                "content": {
                    JSON: {"schema": {"properties": {"model": {"type": "string"}}}}
                }
            }
        },
        "components": {
            "schemas": {
                "Cycle": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/Cycle"},
                        {"$ref": "#/components/schemas/Cycle"},
                    ]
                }
            }
        },
    }
    async with (
        _serve_schema(schema) as (engine, _),
        aiohttp.ClientSession() as client,
    ):
        discovery = OpenAPIInferenceDiscovery()
        assert await discovery.eligible_endpoints(
            client, [engine], "POST", "/task", JSON
        )
        assert (
            await discovery.eligible_endpoints(
                client, [engine], "POST", "/cyclic", JSON
            )
            == []
        )


async def test_oversized_schema_is_cached_as_unavailable(monkeypatch):
    from vllm_router.services.request_service import openapi

    monkeypatch.setattr(openapi, "_SCHEMA_SIZE_LIMIT", 128)
    async with (
        _serve_schema(_upstream_schema()) as (engine, calls),
        aiohttp.ClientSession() as client,
    ):
        discovery = openapi.OpenAPIInferenceDiscovery()
        for _ in range(2):
            assert (
                await discovery.eligible_endpoints(
                    client, [engine], "POST", "/inference/unknown-to-router", JSON
                )
                == []
            )
        assert calls == ["/openapi.json"]
