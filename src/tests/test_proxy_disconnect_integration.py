import asyncio
import json
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import pytest
import uvicorn
from aiohttp import web
from fastapi import FastAPI

from vllm_router.external_providers.models import ExternalProviderConfig
from vllm_router.external_providers.openai_provider import OpenAIProvider
from vllm_router.routers.main_router import main_router
from vllm_router.routers.routing_logic import RoutingLogic, initialize_routing_logic
from vllm_router.service_discovery import EndpointInfo
from vllm_router.stats import request_stats
from vllm_router.utils import SingletonABCMeta

MODEL = "disconnect-test-model"
CHAT_PATH = "/v1/chat/completions"
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
FIRST_CHUNK = b"data: first token\n\n"
WAIT_SECONDS = 3


class NotifyingServer(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self.ready = asyncio.Event()

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        self.ready.set()


@asynccontextmanager
async def serve_router(app):
    config = uvicorn.Config(
        app,
        http="h11",
        lifespan="off",
        access_log=False,
        log_config=None,
        timeout_graceful_shutdown=1,
    )
    server = NotifyingServer(config)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(server.ready.wait(), WAIT_SECONDS)
            assert server.started
            yield port, server
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, WAIT_SECONDS)


@asynccontextmanager
async def serve_backend(path, handler):
    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=1)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        address, port = runner.addresses[0]
        yield f"http://{address}:{port}"
    finally:
        await runner.cleanup()


def configure_discovery(monkeypatch, origins):
    endpoints = [
        EndpointInfo(
            url=url,
            model_names=[MODEL],
            Id=label,
            added_timestamp=0,
            model_label=label,
            sleep=False,
        )
        for label, url in origins.items()
    ]
    monkeypatch.setattr(
        "vllm_router.services.request_service.request.get_service_discovery",
        lambda: SimpleNamespace(get_endpoint_info=lambda: endpoints),
    )


def request_bytes(path, *, stream, multipart):
    if multipart:
        boundary = "disconnect-test-boundary"
        content_type = f"multipart/form-data; boundary={boundary}"
        fields = {"model": MODEL, "stream": str(stream).lower()}
        form = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
            f"\r\n{value}\r\n"
            for name, value in fields.items()
        ]
        form.append(
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\ntest audio\r\n"
            f"--{boundary}--\r\n"
        )
        body = "".join(form).encode()
    else:
        content_type = "application/json"
        body = json.dumps({"model": MODEL, "stream": stream, "max_tokens": 5}).encode()
    headers = (
        f"POST {path} HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    )
    return headers.encode() + body


@pytest.mark.parametrize(
    "mode, phase",
    [
        (mode, phase)
        for mode in ("json", "multipart", "external", "disaggregated", "orchestrated")
        for phase in ("before-headers", "idle-stream")
    ]
    + [("orchestrated", "decode-body")],
)
async def test_client_disconnect_closes_inference_backend(monkeypatch, mode, phase):
    stream = phase == "idle-stream"
    decode_body = phase == "decode-body"
    multipart = mode == "multipart"
    disaggregated = mode in {"disaggregated", "orchestrated"}
    decoding = disaggregated and (stream or decode_body)
    path = TRANSCRIPTIONS_PATH if multipart else CHAT_PATH
    calls = []
    upstream_started = asyncio.Event()
    upstream_closed = asyncio.Event()
    decode_headers_read = asyncio.Event()
    release_backend = asyncio.Event()

    async def stall(request):
        await request.read()
        calls.append("inference")
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        if stream:
            await response.prepare(request)
            await response.write(FIRST_CHUNK)
        elif decode_body:
            response.content_type = "application/json"
            await response.prepare(request)
            await response.write(b'{"choices": [')
        upstream_started.set()
        try:
            await release_backend.wait()
        except asyncio.CancelledError:
            upstream_closed.set()
            raise
        return (
            response if stream or decode_body else web.json_response({"released": True})
        )

    async def prefill(request):
        payload = await request.json()
        assert not payload["stream"]
        calls.append("prefill")
        return web.json_response({"kv_transfer_params": {"do_remote_decode": True}})

    async def unexpected_decode(request):
        calls.append("unexpected decode")
        return web.json_response({"unexpected": True})

    monkeypatch.setattr(SingletonABCMeta, "_instances", {})
    monkeypatch.setattr(request_stats.SingletonMeta, "_instances", {})
    monitor = request_stats.RequestStatsMonitor(10)
    monitor.on_request_complete = Mock(wraps=monitor.on_request_complete)

    app = FastAPI()
    app.include_router(main_router)
    app.state.request_stats_monitor = monitor
    app.state.engine_stats_scraper = SimpleNamespace(get_engine_stats=dict)
    app.state.otel_enabled = False
    app.state.semantic_cache_available = False

    routing_logic = {
        "disaggregated": RoutingLogic.DISAGGREGATED_PREFILL,
        "orchestrated": RoutingLogic.DISAGGREGATED_PREFILL_ORCHESTRATED,
    }.get(mode, RoutingLogic.ROUND_ROBIN)
    app.state.router = initialize_routing_logic(
        routing_logic,
        max_instance_failover_reroute_attempts=1,
        prefill_model_labels=["prefill"],
        decode_model_labels=["decode"],
    )

    async with AsyncExitStack() as stack:
        # Both origins can stall, so round-robin ordering does not affect the test.
        if disaggregated:
            origins = {
                "prefill": await stack.enter_async_context(
                    serve_backend(path, prefill if decoding else stall)
                ),
                "decode": await stack.enter_async_context(
                    serve_backend(path, stall if decoding else unexpected_decode)
                ),
            }
        else:
            origins = {
                label: await stack.enter_async_context(serve_backend(path, stall))
                for label in ("first", "second")
            }
        configure_discovery(monkeypatch, origins)
        trace = aiohttp.TraceConfig()

        async def trace_headers_read(session, context, params):
            if decode_body and str(params.url.origin()) == origins["decode"]:
                decode_headers_read.set()

        trace.on_request_end.append(trace_headers_read)
        session = await stack.enter_async_context(
            aiohttp.ClientSession(trace_configs=[trace])
        )
        app.state.aiohttp_client_wrapper = lambda: session
        if mode == "disaggregated":
            for label, url in origins.items():
                client = await stack.enter_async_context(
                    aiohttp.ClientSession(base_url=url)
                )
                setattr(app.state, f"{label}_client", client)
        if mode == "external":
            monkeypatch.setenv("DISCONNECT_TEST_API_KEY", "local-test-key")
            provider = OpenAIProvider(
                ExternalProviderConfig(
                    name="test-provider",
                    type="openai",
                    api_base=origins["first"],
                    api_key_env_var="DISCONNECT_TEST_API_KEY",
                    max_retries=1,
                )
            )
            stack.push_async_callback(provider.close)
            app.state.external_provider_registry = SimpleNamespace(
                is_external_model=lambda model: model == MODEL,
                lookup_adapter=lambda _: provider,
                get_provider_name=lambda _: "test-provider",
                get_canonical_model_id=lambda _: MODEL,
            )

        port, server = await stack.enter_async_context(serve_router(app))
        stack.callback(release_backend.set)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(request_bytes(path, stream=stream, multipart=multipart))
            await asyncio.wait_for(writer.drain(), WAIT_SECONDS)
            await asyncio.wait_for(upstream_started.wait(), WAIT_SECONDS)
            if stream:
                received = await asyncio.wait_for(
                    reader.readuntil(FIRST_CHUNK), WAIT_SECONDS
                )
                assert received.startswith(b"HTTP/1.1 200 OK\r\n")
            elif decode_body:
                await asyncio.wait_for(decode_headers_read.wait(), WAIT_SECONDS)
            (request_task,) = server.server_state.tasks
        finally:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), WAIT_SECONDS)

        # aiohttp cancels the handler only when the router's upstream socket closes.
        await asyncio.wait_for(upstream_closed.wait(), WAIT_SECONDS)
        await asyncio.wait_for(asyncio.shield(request_task), WAIT_SECONDS)
        assert calls == (["prefill", "inference"] if decoding else ["inference"])
        if mode in {"json", "multipart"}:
            monitor.on_request_complete.assert_called_once()
            assert sum(monitor.in_prefill_requests.values()) == 0
            assert sum(monitor.in_decoding_requests.values()) == 0
            assert sum(monitor.finished_requests.values()) == 1
