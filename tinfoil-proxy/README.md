# Tinfoil Proxy

An OpenAI-compatible API proxy server for Tinfoil's secure enclave models.

## Features

- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`)
- Support for Tinfoil models:
  - `deepseek-r1-70b` - High-performance reasoning model
  - `llama3-3-70b` - Multilingual dialogue model
  - `nomic-embed-text` - Text embeddings
- SSE streaming support for real-time responses
- Full compatibility with OpenAI client libraries

## Setup

### Development/Testing

1. Enter the Nix development shell:
```bash
nix develop
```

2. Build the binary (if not already built):
```bash
just build-tinfoil-proxy
```

3. Set your Tinfoil API key:
```bash
export TINFOIL_API_KEY="your-api-key-here"
```

4. Run the server using the alias:
```bash
tinfoil-proxy
```

Or run from Python directly:
```bash
cd tinfoil-proxy
python tinfoil_proxy.py
```

The server will start on `http://localhost:8093` by default. You can override the port with:
```bash
TINFOIL_PROXY_PORT=8000 tinfoil-proxy
```

### Production (Docker/EIF)

The binary is automatically included in the Docker image and EIF builds. It runs at `/app/tinfoil-proxy` with all necessary libraries and SSL certificates configured.

## Usage

### With OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8093/v1",
    api_key="dummy"  # API key is handled by the proxy
)

response = client.chat.completions.create(
    model="deepseek-r1-70b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### With cURL:

```bash
curl http://localhost:8093/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### List available models:

```bash
curl http://localhost:8093/v1/models
```

## API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completions (streaming and non-streaming)
- `GET /health` - Health check endpoint

## Environment Variables

- `TINFOIL_API_KEY` (required) - Your Tinfoil API key
- `TINFOIL_BASE_URL` (optional) - Override the Tinfoil API base URL

## Integration with OpenSecret

This proxy is designed to work alongside the OpenSecret backend, providing an additional secure LLM provider option through Tinfoil's enclave-based inference.