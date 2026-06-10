# opencode_free Provider Auth Fix

## Problem

The `opencode_free` provider (the free tier of the OpenCode Zen API at
`https://opencode.ai/zen/v1`) does not require an API key.  Requests must be
sent **without** an `Authorization` header.  However, the upstream provider
class `OpenCodeProvider` extends `OpenAIChatTransport`, which uses the OpenAI
Python SDK (`openai` ≥2.37.0) to make HTTP calls.  The SDK has two validation
layers that fight against sending a no-auth request:

1. **Constructor credential check** – `AsyncOpenAI(api_key=...)` raises
   `OpenAIError("Missing credentials...")` when `api_key` is empty/`None`.

2. **Bearer auth injection** – Even when a non-empty placeholder key passes
   the constructor, the overridden `_bearer_auth` property (in
   `_client.py`) unconditionally returns
   `{"Authorization": f"Bearer {api_key}"}` for any truthy key — it does
   **not** check the `WORKLOAD_IDENTITY_API_KEY_PLACEHOLDER` sentinel like
   `auth_headers` does.  This means any non-empty key gets sent as an
   `Authorization` header, which the Zen API rejects for free-tier models.

## Discovery / Root-Cause Analysis

1. **Failed: `static_credential=""`** – SDK constructor rejected empty key.

2. **Failed: `static_credential="workload-identity-auth"`** – Constructor
   accepted the key, and the `auth_headers` property correctly returned `{}`
   for the placeholder.  But the override `_bearer_auth` property does **not**
   check the placeholder — only `auth_headers` does.  The request went out
   with `Authorization: Bearer workload-identity-auth`, which the Zen API
   rejected.

3. **Working: `_NoAuthTransport`** – The root issue is that the openai SDK
   has multiple auth injection points (`auth_headers`, `_bearer_auth`,
   `_auth_headers` with security options), and we cannot control all of them
   from the outside.  Instead of fighting the SDK, we intercept at the HTTP
   transport layer: strip any `Authorization` header just before the bytes
   leave the process.

## Solution

Two changes, one backward-compatible:

### 1. `providers/openai_compat.py` — Optional `http_client` parameter

Added `http_client: httpx.AsyncClient | None = None` to
`OpenAIChatTransport.__init__`.  When provided, this client is passed
directly to `AsyncOpenAI(..., http_client=http_client)`.  When `None`
(default), the existing proxy-based client creation is used.  No existing
subclass is affected.

### 2. `providers/opencode/client.py` — Custom transport strips auth

```python
class _NoAuthTransport(httpx.AsyncHTTPTransport):
    """Transport that strips Authorization headers before sending."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.headers.pop("Authorization", None)
        return await super().handle_async_request(request)
```

`OpenCodeProvider.__init__` now creates an `httpx.AsyncClient` with this
transport and passes it as `http_client` to the parent.  The `api_key`
defaults to `"no-auth"` — a value that satisfies the SDK constructor but is
stripped by the transport before the wire.

### 3. `config/provider_catalog.py` — Static credential

The `opencode_free` provider descriptor uses `static_credential="free"` so
that `_credential_for()` returns a non-empty value, which satisfies
`_require_credential()`.  The actual value is irrelevant because the
transport strips it.

## Files Changed

| File | Change |
|------|--------|
| `providers/openai_compat.py` | Added `http_client` param to `OpenAIChatTransport.__init__` |
| `providers/opencode/client.py` | Added `_NoAuthTransport` and custom `http_client` |
| `config/provider_catalog.py` | Added `opencode_free` provider descriptor with `static_credential="free"` |
| `providers/registry.py` | Registered `OpenCodeProvider` |
| `providers/opencode/__init__.py` | Package init for opencode module |
| `providers/defaults.py` | Default base URL for opencode_free |
| `config/settings.py` | Settings for opencode_free proxy |
| `api/admin_config.py` | Admin UI config for opencode_free |
| `api/admin_static/admin.js` | Admin JS for opencode_free |
| `pyproject.toml` | Version bump |
| `tests/contracts/test_provider_catalog_order.py` | Test update for new provider |
| `uv.lock` | Lock file update |

## Verification

Start the server and test:

```bash
fcc-server --debug
curl -s --max-time 30 \
  -X POST http://127.0.0.1:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: <your-auth-token>" \
  -d '{
    "model":"anthropic/opencode_free/deepseek-v4-flash-free",
    "max_tokens":50,
    "messages":[{"role":"user","content":"Say hello in 3 words"}]
  }'
```

A valid SSE stream with `message_start`, thinking blocks, text, and
`message_stop` confirms the fix.
