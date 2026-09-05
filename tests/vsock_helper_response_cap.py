#!/usr/bin/env python3

import importlib.util
import io
import json
import os
from contextlib import redirect_stderr
from pathlib import Path


helper_path = Path(os.environ["VSOCK_HELPER_UNDER_TEST"])
spec = importlib.util.spec_from_file_location("vsock_helper_under_test", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {helper_path}")

helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
if not hasattr(helper.socket, "AF_VSOCK"):
    helper.socket.AF_VSOCK = 40
assert helper.MAX_RESPONSE_BYTES == 1024 * 1024


class ChunkedResponseSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def close(self):
        pass

    def connect(self, _address):
        pass

    def recv(self, length):
        if not self.chunks:
            raise AssertionError("helper read beyond the fixture response")
        chunk = self.chunks.pop(0)
        assert len(chunk) <= length
        return chunk

    def sendall(self, _request):
        pass

    def settimeout(self, _seconds):
        pass


def chunks_for(payload):
    chunk_size = 4096
    return [
        payload[offset : offset + chunk_size]
        for offset in range(0, len(payload), chunk_size)
    ] + [b""]


def request_with_payload(payload):
    fake_socket = ChunkedResponseSocket(chunks_for(payload))
    socket_attempts = 0

    def socket_factory(*_args, **_kwargs):
        nonlocal socket_attempts
        socket_attempts += 1
        return fake_socket

    helper.socket.socket = socket_factory
    helper.select.select = lambda *_args, **_kwargs: ([fake_socket], [], [])
    helper.time.sleep = lambda _seconds: None
    with redirect_stderr(io.StringIO()):
        response = helper.vsock_request(
            3,
            8003,
            "{}",
            max_retries=3,
            retry_delay=0,
            initial_delay=0,
        )
    return response, fake_socket, socket_attempts


json_prefix = b'{"ok":"'
json_suffix = b'"}'
json_overhead = len(json_prefix) + len(json_suffix)

small_payload = json_prefix + b"x" * 8192 + json_suffix
small_response, small_socket, small_attempts = request_with_payload(small_payload)
assert json.loads(small_response) == {"ok": "x" * 8192}
assert not small_socket.chunks
assert small_attempts == 1

at_limit_payload = (
    json_prefix
    + b"x" * (helper.MAX_RESPONSE_BYTES - json_overhead)
    + json_suffix
)
assert len(at_limit_payload) == helper.MAX_RESPONSE_BYTES
at_limit_response, at_limit_socket, at_limit_attempts = request_with_payload(
    at_limit_payload
)
assert len(at_limit_response.encode()) == helper.MAX_RESPONSE_BYTES
assert len(json.loads(at_limit_response)["ok"]) == (
    helper.MAX_RESPONSE_BYTES - json_overhead
)
assert not at_limit_socket.chunks
assert at_limit_attempts == 1

over_limit_payload = (
    json_prefix
    + b"x" * (helper.MAX_RESPONSE_BYTES + 1 - json_overhead)
    + json_suffix
)
assert len(over_limit_payload) == helper.MAX_RESPONSE_BYTES + 1
over_limit_response, over_limit_socket, over_limit_attempts = request_with_payload(
    over_limit_payload
)
assert json.loads(over_limit_response) == {
    "error": "VSOCK response exceeded size limit"
}
# The helper rejects the byte that crosses the limit without waiting for EOF.
assert over_limit_socket.chunks == [b""]
assert over_limit_attempts == 1

print("vsock helper response-cap boundary tests passed")
