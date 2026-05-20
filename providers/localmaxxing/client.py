"""LocalMaxxing provider using OpenAI-compatible chat completions."""

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.base import ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.openai_compat import OpenAIChatTransport


class LocalMaxxingProvider(OpenAIChatTransport):
    """LocalMaxxing provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LocalMaxxing",
            base_url=config.base_url or "",
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build OpenAI-format request body from Anthropic request for LocalMaxxing."""
        if thinking_enabled is None:
            thinking_enabled = self._is_thinking_enabled(request)
        logger.debug(
            "LOCALMAXXING_REQUEST: conversion start model={} msgs={}",
            getattr(request, "model", "?"),
            len(getattr(request, "messages", [])),
        )
        try:
            body = build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

        extra_body: dict[str, Any] = {}
        request_extra = getattr(request, "extra_body", None)
        if request_extra:
            extra_body.update(request_extra)

        if extra_body:
            body["extra_body"] = extra_body

        logger.debug(
            "LOCALMAXXING_REQUEST: conversion done model={} msgs={} tools={}",
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )
        return body
