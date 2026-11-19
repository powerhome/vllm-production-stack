# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack

from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from vllm_router.log import init_logger

logger = init_logger(__name__)
_LIFETIME_KEY = "vllm_router.request_lifetime"


class RequestLifetime:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.disconnected = False
        self._cleanup = AsyncExitStack()

    def add_cleanup(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register resource cleanup after the response handler has stopped."""
        self._cleanup.push_async_callback(callback)

    async def close(self) -> None:
        await self._cleanup.aclose()


def request_lifetime(request: Request) -> RequestLifetime | None:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        lifetime = scope.get(_LIFETIME_KEY)
        if isinstance(lifetime, RequestLifetime):
            return lifetime
    return None


class DisconnectAwareRoute(APIRoute):
    """Cancel request-scoped forwarding as soon as the client disconnects."""

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        request_id = Headers(scope=scope).get("x-request-id") or str(uuid.uuid4())
        lifetime = RequestLifetime(request_id)
        scope[_LIFETIME_KEY] = lifetime
        frames: asyncio.Queue[Message] = asyncio.Queue(maxsize=1)
        peer_disconnected = asyncio.get_running_loop().create_future()
        response_started = False
        response_completed = False

        def mark_disconnected() -> None:
            if not response_completed and not lifetime.disconnected:
                lifetime.disconnected = True
                peer_disconnected.set_result(None)

        async def relay_receive() -> None:
            # Keep just one queued upload frame. The receive owner can detect a
            # disconnect even when the endpoint is waiting on upstream I/O.
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    # Uvicorn returns this event after a successful final send too.
                    mark_disconnected()
                    return
                await frames.put(message)

        async def receive_for_app() -> Message:
            if lifetime.disconnected or response_completed:
                return {"type": "http.disconnect"}
            return await frames.get()

        async def track_send(message: Message) -> None:
            nonlocal response_started, response_completed
            if lifetime.disconnected:
                raise ClientDisconnect()
            try:
                await send(message)
            except OSError:
                mark_disconnected()
                raise
            if message["type"] == "http.response.start":
                response_started = True
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_completed = True

        receiver = asyncio.create_task(relay_receive())
        handler = asyncio.create_task(
            super().handle(scope, receive_for_app, track_send)
        )
        try:
            done, _ = await asyncio.wait(
                (receiver, handler, peer_disconnected),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receiver in done:
                receiver.result()
            if not lifetime.disconnected:
                await handler
        except (ClientDisconnect, OSError):
            if not lifetime.disconnected:
                raise
        finally:
            for task in (receiver, handler):
                if not task.done():
                    task.cancel()
            await asyncio.gather(receiver, handler, return_exceptions=True)
            peer_disconnected.cancel()
            try:
                await lifetime.close()
            except Exception:
                logger.exception("Failed to clean up request %s", request_id)

        if lifetime.disconnected:
            if response_started:
                logger.info("Request %s cancelled during the response", request_id)
                return
            logger.info(
                "Request %s cancelled before response headers (499)", request_id
            )
            response = Response(status_code=499, headers={"X-Request-Id": request_id})
            try:
                await response(scope, receive_for_app, send)
            except OSError:
                pass
