# Tinfoil Proxy

An OpenAI-compatible API proxy server for Tinfoil's secure enclave models, written in Go.

## Features

- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`)
- Support for streaming and non-streaming chat completions
- Secure communication with Tinfoil enclaves
- High-performance Go implementation

## Building

### Reproducible Linux binaries

```bash
./build.sh
```

The script enters the nested Nix development shell and uses its pinned Go
toolchain to build both checked-in static Linux binaries:

- `dist/tinfoil-proxy` (`linux/arm64`)
- `dist/tinfoil-proxy-x86_64` (`linux/amd64`)

### Building directly

```bash
go mod download
CGO_ENABLED=0 go build -o tinfoil-proxy .
```

## Checks

Run the tests, static analysis, and Go vulnerability scan with the same pinned
toolchain:

```bash
nix develop -c go test ./...
nix develop -c go vet ./...
nix develop -c govulncheck ./...
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
    model="llama3-3-70b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```
