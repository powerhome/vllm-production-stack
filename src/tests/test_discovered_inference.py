import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import FastAPI

from vllm_router.routers import main_router as routes
from vllm_router.routers.routing_logic import PriorityRouter, RoundRobinRouter
from vllm_router.services.request_service import request as forwarding


@asynccontextmanager
async def proxy_client(monkeypatch, *backends):
    async with AsyncExitStack() as stack:
        endpoints = []
        for index, backend in enumerate(backends):
            server = await stack.enter_async_context(TestServer(backend))
            endpoints.append(
                SimpleNamespace(
                    Id=str(index),
                    url=str(server.make_url("/")).rstrip("/"),
                    model_names=["test-model"],
                    sleep=False,
                )
            )
        discovery = SimpleNamespace(
            get_endpoint_info=lambda: endpoints,
            has_ever_seen_model=lambda _: True,
            aliases={},
        )
        monkeypatch.setattr(forwarding, "get_service_discovery", lambda: discovery)

        app = FastAPI()
        app.include_router(routes.main_router)
        app.include_router(routes.inference_fallback_router)
        session = await stack.enter_async_context(aiohttp.ClientSession())
        app.state.aiohttp_client_wrapper = lambda: session
        app.state.engine_stats_scraper = Mock()
        app.state.request_stats_monitor = Mock()
        app.state.otel_enabled = False
        app.state.semantic_cache_available = False
        app.state.router = Mock(spec=RoundRobinRouter)
        app.state.router.max_instance_failover_reroute_attempts = 0
        app.state.router.route_request.side_effect = lambda candidates, *_: (
            candidates[0].url
        )
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://router"
            )
        )
        yield app, client, endpoints


@pytest.mark.asyncio
async def test_schema_advertised_inference_routes_to_capable_engine(monkeypatch):
    path = "/future/classify"
    schema = {
        "openapi": "3.1.0",
        "paths": {
            path: {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"type": "string"},
                                        "input": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    received = []

    async def classify(request):
        received.append((list(request.query.items()), await request.json()))
        return web.json_response({"label": "positive"}, status=201)

    async def capable_schema(_):
        return web.json_response(schema)

    async def other_schema(_):
        return web.json_response({"openapi": "3.1.0", "paths": {}})

    capable = web.Application()
    capable.router.add_get("/openapi.json", capable_schema)
    capable.router.add_post(path, classify)
    other = web.Application()
    other.router.add_get("/openapi.json", other_schema)
    other.router.add_post(path, classify)

    async with proxy_client(monkeypatch, other, capable) as (app, client, endpoints):
        response = await client.post(
            path,
            params=[("format", "first"), ("format", "second")],
            json={"input": "text"},
        )
        assert response.status_code == 201
        assert response.json() == {"label": "positive"}
        assert received == [
            ([("format", "first"), ("format", "second")], {"input": "text"})
        ]
        assert app.state.router.route_request.call_args.args[0] == [endpoints[1]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,expected_priority", [({"x-request-priority": "-3"}, "-3"), ({}, "5")]
)
async def test_explicit_engine_selection_resolves_multipart_priority(
    monkeypatch, headers, expected_priority
):
    path = "/future/audio"
    schema = {
        "openapi": "3.1.0",
        "paths": {
            path: {
                "post": {
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"model": {"type": "string"}},
                                    "required": ["model"],
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    def backend(number):
        async def openapi(_):
            return web.json_response(schema)

        async def infer(request):
            assert "id" not in request.query
            form = await request.post()
            return web.json_response({"engine": number, "priority": form["priority"]})

        app = web.Application()
        app.router.add_get("/openapi.json", openapi)
        app.router.add_post(path, infer)
        return app

    class TestPriorityRouter(PriorityRouter):
        pass

    async with proxy_client(monkeypatch, backend(0), backend(1)) as (app, client, _):
        app.state.router = TestPriorityRouter(priority_default=5)
        response = await client.post(
            path,
            params={"id": "1"},
            headers=headers,
            data={"model": "test-model"},
            files={"file": ("test.wav", b"audio", "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json() == {"engine": 1, "priority": expected_priority}


@pytest.mark.asyncio
async def test_responses_followups_only_reach_confirmed_owner(monkeypatch):
    parameter = {"name": "response_id", "in": "path", "required": True}
    schema = {
        "openapi": "3.1.0",
        "paths": {
            "/v1/responses/{response_id}": {
                "get": {
                    "operationId": "retrieve_responses_v1_responses__response_id__get",
                    "parameters": [parameter],
                },
            },
            "/v1/responses/{response_id}/cancel": {
                "post": {
                    "operationId": "cancel_responses_v1_responses__response_id__cancel_post",
                    "parameters": [parameter],
                },
            },
        },
    }
    events = []

    def backend(has_response):
        async def openapi(_):
            return web.json_response(schema)

        async def retrieve(request):
            assert request.headers.get("Authorization") == "Bearer test-client"
            if not has_response:
                return web.Response(status=404)
            if request.query.get("stream") == "true":
                events.append("stream")
                assert request.query["starting_after"] == "2"
                return web.Response(
                    body=b"data: {}\n\n", content_type="text/event-stream"
                )
            return web.json_response({"id": "resp_example", "object": "response"})

        async def cancel(request):
            events.append(("cancel", has_response))
            assert request.headers.get("Authorization") == "Bearer test-client"
            return web.json_response({"status": "cancelled"})

        async def create(request):
            events.append(("continue", has_response))
            assert (await request.json())["previous_response_id"] == "resp_example"
            return web.json_response({"id": "resp_next", "object": "response"})

        app = web.Application()
        app.router.add_get("/openapi.json", openapi)
        app.router.add_get("/v1/responses/{response_id}", retrieve)
        app.router.add_post("/v1/responses/{response_id}/cancel", cancel)
        app.router.add_post("/v1/responses", create)
        return app

    async with proxy_client(monkeypatch, backend(False), backend(True)) as (
        _,
        client,
        endpoints,
    ):
        headers = {"Authorization": "Bearer test-client"}
        endpoints[1].sleep = True
        retrieval = await client.get(
            "/v1/responses/resp_example?stream=true&starting_after=2", headers=headers
        )
        assert retrieval.status_code == 200
        assert retrieval.content == b"data: {}\n\n"
        cancelled = await client.post(
            "/v1/responses/resp_example/cancel", headers=headers
        )
        assert cancelled.json() == {"status": "cancelled"}
        assert (
            await client.get("/v1/responses/resp_example/cancel", headers=headers)
        ).status_code == 404
        assert (
            await client.post("/v1/responses/resp_example", headers=headers)
        ).status_code == 404
        continuation_payload = {
            "previous_response_id": "resp_example",
            "input": "next",
        }
        asleep = await client.post(
            "/v1/responses", headers=headers, json=continuation_payload
        )
        assert asleep.status_code == 503
        endpoints[1].sleep = False
        continuation = await client.post(
            "/v1/responses", headers=headers, json=continuation_payload
        )
        assert continuation.json()["id"] == "resp_next"
        assert events == ["stream", ("cancel", True), ("continue", True)]


@pytest.mark.asyncio
async def test_non_streaming_timeout_can_expose_504_after_upstream_headers(monkeypatch):
    upstream_started = asyncio.Event()
    finish = asyncio.Event()

    async def stall_body(request):
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        upstream_started.set()
        await finish.wait()
        return response

    backend = web.Application()
    backend.router.add_post("/v1/chat/completions", stall_body)
    async with proxy_client(monkeypatch, backend) as (app, client, _):
        app.state.backend_request_timeout_seconds = 0.2
        try:
            response = await client.post(
                "/v1/chat/completions", json={"model": "test-model", "stream": False}
            )
            assert upstream_started.is_set()
            assert response.status_code == 504
        finally:
            finish.set()
