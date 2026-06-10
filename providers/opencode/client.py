"""OpenCode Zen provider implementation (OpenAI-compatible Chat Completions)."""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import ProviderConfig
from providers.defaults import OPENCODE_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class _NoAuthTransport(httpx.AsyncHTTPTransport):
    """Transport that strips Authorization headers before sending."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.headers.pop("Authorization", None)
        return await super().handle_async_request(request)


class OpenCodeProvider(OpenAIChatTransport):
    """OpenCode Zen provider using ``https://opencode.ai/zen/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig, provider_name: str = "OPENCODE"):
        http_client = httpx.AsyncClient(
            transport=_NoAuthTransport(),
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )
        super().__init__(
            config,
            provider_name=provider_name,
            base_url=config.base_url or OPENCODE_DEFAULT_BASE,
            api_key=config.api_key or "no-auth",
            http_client=http_client,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
