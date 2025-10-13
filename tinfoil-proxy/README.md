# Tinfoil Proxy

An OpenAI-compatible API proxy server for Tinfoil's secure enclave models, written in Go.

## Features

- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`)
- Support for streaming and non-streaming chat completions
- Secure communication with Tinfoil enclaves
- High-performance Go implementation

## Building

### Using Docker (recommended)

```bash
./build.sh
```

This will build the Docker image and extract the binary to `dist/tinfoil-proxy`.

### Building directly

```bash
go mod download
CGO_ENABLED=0 go build -o tinfoil-proxy .
```

## Running

Set the required environment variables:

```bash
export TINFOIL_API_KEY=your-api-key
export TINFOIL_PROXY_PORT=8093  # optional, defaults to 8093
```

Run the proxy:

```bash
./dist/tinfoil-proxy
```

## Supported Models

- `deepseek-r1-0528` - High-performance reasoning model
- `deepseek-v31-terminus` - Advanced reasoning model (V3.1 Terminus)
- `llama3-3-70b` - Multilingual model optimized for dialogue
- `nomic-embed-text` - Text embedding model

## API Usage

The proxy provides an OpenAI-compatible API. You can use any OpenAI client library by pointing it to the proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8093/v1",
    api_key="dummy"  # The actual API key is set via TINFOIL_API_KEY env var
)

response = client.chat.completions.create(
    model="deepseek-r1-0528",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```