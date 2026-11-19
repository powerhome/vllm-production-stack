import asyncio
import socket
from contextlib import asynccontextmanager

import aiohttp
import pytest
import uvicorn
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from vllm_router.services.request_service.disconnect import (
    DisconnectAwareRoute,
    request_lifetime,
)


@asynccontextmanager
async def serve(router, spec_version=None):
    app = FastAPI()
    app.include_router(router)
    started = asyncio.Event()

    async def asgi(scope, receive, send):
        if spec_version is not None:
            scope["asgi"]["spec_version"] = spec_version
        await app(scope, receive, send)

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            started.set()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        host, port = listener.getsockname()
        server = Server(
            uvicorn.Config(asgi, lifespan="off", timeout_graceful_shutdown=1)
        )
        worker = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(started.wait(), 3)
            yield host, port
        finally:
            server.should_exit = True
            await asyncio.wait_for(worker, 3)


@asynccontextmanager
async def client(address, request_bytes):
    reader, writer = await asyncio.open_connection(*address)
    try:
        writer.write(request_bytes)
        await writer.drain()
        yield reader, writer
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_disconnect_during_upload_preserves_received_frames_and_cleans_up():
    router = APIRouter(route_class=DisconnectAwareRoute)
    started = asyncio.Event()
    cleaned = asyncio.Event()
    received = []

    @router.post("/upload")
    async def upload(request: Request):
        lifetime = request_lifetime(request)

        async def cleanup():
            assert lifetime.disconnected
            cleaned.set()

        lifetime.add_cleanup(cleanup)
        async for chunk in request.stream():
            received.append(chunk)
            started.set()
        return Response()

    async with serve(router) as address:
        async with client(
            address,
            b"POST /upload HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 12\r\n\r\nfirst",
        ):
            await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(cleaned.wait(), 2)
        assert received == [b"first"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "spec_version"), [(False, None), (True, None), (True, "2.4")]
)
async def test_disconnect_closes_upstream_before_headers_or_during_idle_stream(
    stream, spec_version
):
    router = APIRouter(route_class=DisconnectAwareRoute)
    upstream_started = asyncio.Event()
    upstream_closed = asyncio.Event()
    cleanup_order = []
    cleaned = asyncio.Event()

    async def infer(request):
        try:
            if stream:
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(b"data: first\n\n")
            upstream_started.set()
            await asyncio.Future()
        finally:
            upstream_closed.set()

    app = web.Application()
    app.router.add_post("/infer", infer)

    async with (
        TestServer(app, handler_cancellation=True) as upstream,
        aiohttp.ClientSession() as session,
    ):

        @router.post("/infer")
        async def forward(request: Request):
            lifetime = request_lifetime(request)

            async def finish():
                assert lifetime.disconnected
                assert lifetime.request_id == "socket-test"
                cleanup_order.append("finish")
                cleaned.set()

            lifetime.add_cleanup(finish)
            upstream_request = session.post(
                upstream.make_url("/infer"), data=await request.body()
            )
            if not stream:
                async with upstream_request as response:
                    return Response(await response.read())

            response = await upstream_request

            async def close():
                response.close()
                cleanup_order.append("close")

            lifetime.add_cleanup(close)
            return StreamingResponse(response.content.iter_any())

        async with serve(router, spec_version) as address:
            async with client(
                address,
                b"POST /infer HTTP/1.1\r\nHost: localhost\r\n"
                b"X-Request-Id: socket-test\r\n"
                b"Content-Length: 4\r\n\r\nbody",
            ) as (reader, _):
                await asyncio.wait_for(upstream_started.wait(), 2)
                if stream:
                    await asyncio.wait_for(reader.readuntil(b"data: first\n\n"), 2)

            await asyncio.wait_for(upstream_closed.wait(), 2)
            await asyncio.wait_for(cleaned.wait(), 2)
            assert cleanup_order == (["close", "finish"] if stream else ["finish"])


@pytest.mark.asyncio
async def test_completed_stream_keeps_background_callback_and_cleans_up():
    router = APIRouter(route_class=DisconnectAwareRoute)
    finish_callback = asyncio.Event()
    cleaned = asyncio.Event()

    @router.post("/stream")
    async def stream(request: Request):
        lifetime = request_lifetime(request)
        body = await request.body()

        async def cleanup():
            assert not lifetime.disconnected
            assert finish_callback.is_set()
            cleaned.set()

        async def callback():
            await asyncio.sleep(0)
            finish_callback.set()

        lifetime.add_cleanup(cleanup)

        async def generate():
            yield body

        return StreamingResponse(generate(), background=BackgroundTask(callback))

    async with serve(router) as address:
        async with client(
            address,
            b"POST /stream HTTP/1.1\r\nHost: localhost\r\n"
            b"Connection: close\r\nContent-Length: 4\r\n\r\nbody",
        ) as (reader, _):
            response = await asyncio.wait_for(reader.read(), 2)
        assert response.startswith(b"HTTP/1.1 200")
        assert response.endswith(b"4\r\nbody\r\n0\r\n\r\n")
        await asyncio.wait_for(cleaned.wait(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "termination", ["peer", "send_error", "send_error_swallowed", "server"]
)
async def test_before_headers_reports_client_disconnect_only(termination):
    router = APIRouter(route_class=DisconnectAwareRoute)
    app = FastAPI()
    peer_closed = asyncio.Event()
    handler_started = asyncio.Event()
    receive_sent_body = False
    cleaned = []
    sent = []

    class SwallowingResponse(Response):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            except OSError:
                await asyncio.Future()

    @router.get("/work")
    async def work(request: Request):
        lifetime = request_lifetime(request)

        async def cleanup():
            cleaned.append(lifetime.disconnected)

        lifetime.add_cleanup(cleanup)
        handler_started.set()
        if termination == "send_error":
            return Response()
        if termination == "send_error_swallowed":
            return SwallowingResponse()
        await asyncio.Future()

    app.include_router(router)
    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.4"},
        "method": "GET",
        "path": "/work",
        "query_string": b"",
        "headers": [(b"x-request-id", b"test-499")],
    }

    async def receive():
        nonlocal receive_sent_body
        if not receive_sent_body:
            receive_sent_body = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await peer_closed.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if termination.startswith("send_error"):
            raise BrokenPipeError

    worker = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(handler_started.wait(), 2)
        if termination == "server":
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 2)
            assert cleaned == [False]
            assert sent == []
        else:
            if termination == "peer":
                peer_closed.set()
            await asyncio.wait_for(worker, 2)
            assert cleaned == [True]
            expected = [200, 499] if termination.startswith("send_error") else [499]
            assert [msg["status"] for msg in sent if "status" in msg] == expected
            assert (b"x-request-id", b"test-499") in next(
                msg["headers"] for msg in sent if msg.get("status") == 499
            )
    finally:
        if not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
