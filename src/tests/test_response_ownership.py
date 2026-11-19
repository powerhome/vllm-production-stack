import asyncio
from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import HTTPException

from vllm_router.service_discovery import EndpointInfo
from vllm_router.services.request_service import responses


def endpoint(url):
    return EndpointInfo(url, ["test-model"], url, 0, "default", False)


async def test_looks_up_owner_once_per_address_with_caller_authorization():
    response_id = "resp_example:%2F#?"
    hits = Counter()

    def backend(owner):
        async def get_response(request):
            hits[owner] += 1
            assert request.match_info["response_id"] == response_id
            assert request.query == {"stream": "false"}
            assert request.headers["Authorization"] == "Bearer caller"
            assert request.headers["Accept"] == "application/json"
            assert "Range" not in request.headers
            if not owner:
                raise web.HTTPNotFound()
            return web.json_response({"id": response_id, "object": "response"})

        app = web.Application()
        app.router.add_get("/v1/responses/{response_id}", get_response)
        return TestServer(app)

    async with (
        backend(False) as miss,
        backend(True) as owner,
        aiohttp.ClientSession() as client,
    ):
        owner_endpoint = endpoint(str(owner.make_url("")))
        assert (
            await responses.find_response_owner(
                client,
                [
                    endpoint(str(miss.make_url(""))),
                    owner_endpoint,
                    endpoint(owner_endpoint.url + "/"),
                ],
                response_id,
                {"Authorization": "Bearer caller", "Range": "bytes=0-1"},
            )
            is owner_endpoint
        )
    assert hits == {False: 1, True: 1}


@pytest.mark.parametrize(
    ("results", "status"),
    [
        ([404, 404], 404),
        ([200, 200], 409),
        ([200, 200, 503], 409),
        ([200, 503], 503),
        ([404, 503], 503),
        ([404, 401], 401),
        ([404, 403], 403),
        ([404, 401, 503], 503),
        ([200, 403], 503),
        ([], 503),
    ],
)
async def test_owner_lookup_errors_are_deterministic(results, status):
    def backend(code):
        async def get_response(request):
            if code == 200:
                return web.json_response({"id": "lookup", "object": "response"})
            return web.Response(status=code)

        app = web.Application()
        app.router.add_get("/v1/responses/{response_id}", get_response)
        return TestServer(app)

    async with AsyncExitStack() as stack, aiohttp.ClientSession() as client:
        servers = [await stack.enter_async_context(backend(code)) for code in results]
        endpoints = [endpoint(str(server.make_url(""))) for server in servers]
        with pytest.raises(HTTPException) as error:
            await responses.find_response_owner(client, endpoints, "lookup", {})
    assert error.value.status_code == status


@pytest.mark.parametrize(
    "body", [{"id": "wrong", "object": "response"}, {"id": "lookup"}, []]
)
async def test_invalid_success_payload_does_not_prove_ownership(body):
    async def get_response(request):
        return web.json_response(body)

    app = web.Application()
    app.router.add_get("/v1/responses/{response_id}", get_response)
    async with TestServer(app) as server, aiohttp.ClientSession() as client:
        with pytest.raises(HTTPException) as error:
            await responses.find_response_owner(
                client, [endpoint(str(server.make_url("")))], "lookup", {}
            )
    assert error.value.status_code == 503


async def test_owner_lookup_does_not_follow_redirects():
    hits = []

    async def redirect(request):
        raise web.HTTPFound("/other")

    async def redirected(request):
        hits.append(request)
        return web.json_response({"id": "lookup", "object": "response"})

    app = web.Application()
    app.router.add_get("/v1/responses/{response_id}", redirect)
    app.router.add_get("/other", redirected)
    async with TestServer(app) as server, aiohttp.ClientSession() as client:
        with pytest.raises(HTTPException) as error:
            await responses.find_response_owner(
                client, [endpoint(str(server.make_url("")))], "lookup", {}
            )
    assert error.value.status_code == 503
    assert hits == []


@pytest.mark.parametrize("disconnect", [True, False], ids=["disconnect", "deadline"])
async def test_probe_workers_are_bounded_and_joined_on_abort(monkeypatch, disconnect):
    monkeypatch.setattr(responses, "_OWNER_LOOKUP_WORKERS", 2)
    monkeypatch.setattr(responses, "_OWNER_LOOKUP_TIMEOUT", 0.3)
    started = asyncio.Event()
    calls = 0
    active = 0

    class StallingClient:
        @asynccontextmanager
        async def get(self, *_args, **_kwargs):
            nonlocal calls, active
            calls += 1
            active += 1
            if calls == 2:
                started.set()
            try:
                await asyncio.Future()
                yield
            finally:
                active -= 1

    engines = [endpoint(f"http://engine-{index}") for index in range(5)]
    async with asyncio.timeout(2):
        task = asyncio.create_task(
            responses.find_response_owner(StallingClient(), engines, "lookup", {})
        )
        try:
            await started.wait()
            if disconnect:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(HTTPException) as error:
                    await task
                assert error.value.status_code == 503
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert calls == 2
    assert active == 0


@pytest.mark.parametrize("response_id", ["", ".", "..", "a/b", "a\\b", "a\r", "\ud800"])
async def test_unsafe_response_id_cannot_change_probe_target(response_id):
    with pytest.raises(HTTPException) as error:
        await responses.find_response_owner(None, [], response_id, {})
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    ("path", "response_id"),
    [
        ("/v1/responses", None),
        ("/v1/responses/resp_a", "resp_a"),
        ("/v1/responses/resp_a/cancel", "resp_a"),
        ("/v1/responses/resp_a/cancel/more", None),
        ("/v1/responses/resp_a/other", None),
        ("/v1/responses/../cancel", None),
    ],
)
def test_resource_matching(path, response_id):
    assert responses.response_resource_id(path) == response_id
