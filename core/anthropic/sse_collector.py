"""Collect SSE event streams and assemble them into Anthropic Message JSON responses.

Used for non-streaming (JSON) responses where the underlying pipeline emits SSE
internally but the client expects a complete ``Message`` object.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from providers.exceptions import ProviderError

from .stream_contracts import parse_sse_text


async def collect_sse_to_message(stream: AsyncIterator[str]) -> dict:
    """Consume an SSE event stream and produce a ``Message`` JSON dict.

    Parameters
    ----------
    stream:
        Async iterator yielding raw SSE event strings (the same format the
        streaming pipeline produces).

    Returns
    -------
    dict
        An Anthropic ``Message``-shaped JSON dict suitable for returning as a
        non-streaming response body.

    Raises
    ------
    ProviderError
        If the stream contains a top-level ``event: error``.
    ValueError
        If the stream ends without a ``message_start`` event.
    """
    parts = [chunk async for chunk in stream]
    events = parse_sse_text("".join(parts))

    message: dict | None = None
    blocks: dict[int, dict] = {}
    stop_reason: str | None = None
    output_tokens: int = 0

    for ev in events:
        if ev.event == "message_start":
            msg = ev.data["message"]
            message = {
                "id": msg["id"],
                "type": "message",
                "role": msg["role"],
                "content": [],
                "model": msg["model"],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": msg.get("usage", {"input_tokens": 0, "output_tokens": 0}),
            }
        elif ev.event == "content_block_start":
            idx = ev.data["index"]
            blocks[idx] = dict(ev.data["content_block"])
        elif ev.event == "content_block_delta":
            idx = ev.data["index"]
            delta = ev.data["delta"]
            delta_type = delta.get("type", "")
            block = blocks.get(idx)
            if block is None:
                continue
            if delta_type == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta_type == "input_json_delta":
                block.setdefault("_partial_input", "")
                block["_partial_input"] += delta.get("partial_json", "")
            elif delta_type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get(
                    "thinking", ""
                )
            elif delta_type == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get(
                    "signature", ""
                )
        elif ev.event == "content_block_stop":
            idx = ev.data["index"]
            block = blocks.get(idx)
            if block is not None and block.get("type") == "tool_use":
                _finalize_tool_input(block)
        elif ev.event == "message_delta":
            stop_reason = ev.data.get("delta", {}).get("stop_reason")
            output_tokens = ev.data.get("usage", {}).get("output_tokens", 0) or 0
        elif ev.event == "message_stop":
            break
        elif ev.event == "error":
            error_obj = ev.data.get("error", {})
            error_msg = (
                error_obj.get("message")
                if isinstance(error_obj, dict)
                else str(error_obj) or "Unknown provider error"
            )
            raise ProviderError(error_msg)

    if message is None:
        raise ValueError("No message_start event found in SSE stream")

    message["content"] = [blocks[i] for i in sorted(blocks)]
    message["stop_reason"] = stop_reason
    output_usage = message.setdefault("usage", {})
    if isinstance(output_tokens, int | str):
        try:
            output_usage["output_tokens"] = int(output_tokens)
        except TypeError, ValueError:
            output_usage["output_tokens"] = 0

    return message


def _finalize_tool_input(block: dict) -> None:
    """Parse accumulated ``input_json_delta`` fragments into ``input``."""
    raw = block.pop("_partial_input", None)
    if raw is None:
        return
    try:
        block["input"] = json.loads(raw)
    except json.JSONDecodeError:
        block["input"] = raw
