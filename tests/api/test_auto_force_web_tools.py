"""Tests for auto-forcing Anthropic server tools on OpenAI Chat upstreams."""

import json


def test_auto_force_web_search_tool_on_openai_chat_upstream(
    test_client, test_provider
):
    """When ENABLE_WEB_SERVER_TOOLS=True, a request that lists web_search without forcing
    should automatically have tool_choice set to web_search for OpenAI Chat upstreams.
    """
    from api.models.anthropic import MessagesRequest, Tool

    settings = test_client.app.state.settings
    settings.enable_web_server_tools = True

    request = MessagesRequest(
        model="nvidia_nim/fake-model",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "search the web for claude code"}],
            }
        ],
        tools=[
            Tool(name="web_search", type="web_search_20250305"),
        ],
    )
    response = test_client.post("/v1/messages", content=request.model_dump_json())
    assert response.status_code == 200

    # Verify the SSE stream contains server_tool_use and web_search_tool_result
    events = list(response.iter_lines())
    event_names = [
        line.decode().strip() if isinstance(line, bytes) else line.strip()
        for line in events
        if line
    ]
    data_blocks = [
        json.loads(line.split(":", 1)[1])
        for line in event_names
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_use_events = [
        b
        for b in data_blocks
        if b.get("type") == "content_block_start"
        and b.get("content_block", {}).get("type") == "server_tool_use"
    ]
    tool_result_events = [
        b
        for b in data_blocks
        if b.get("type") == "content_block_start"
        and b.get("content_block", {}).get("type") in ("web_search_tool_result", "web_fetch_tool_result")
    ]
    assert len(tool_use_events) >= 1, "Expected at least one server_tool_use event"
    assert len(tool_result_events) >= 1, "Expected at least one web_search_tool_result event"