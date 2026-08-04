#!/usr/bin/env python3

import importlib.util
import json
import os
from pathlib import Path


helper_path = Path(os.environ["VSOCK_HELPER_UNDER_TEST"])
spec = importlib.util.spec_from_file_location("vsock_helper_under_test", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {helper_path}")

helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
if not hasattr(helper.socket, "AF_VSOCK"):
    helper.socket.AF_VSOCK = 40


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def close(self):
        pass

    def connect(self, _address):
        pass

    def recv(self, _length):
        return self.chunks.pop(0)

    def sendall(self, _request):
        pass

    def settimeout(self, _seconds):
        pass


def request_with_chunks(chunks):
    fake_socket = FakeSocket(chunks)
    helper.socket.socket = lambda *_args, **_kwargs: fake_socket
    helper.select.select = lambda *_args, **_kwargs: ([fake_socket], [], [])
    helper.time.sleep = lambda _seconds: None
    return helper.vsock_request(
        3,
        8003,
        "{}",
        max_retries=1,
        retry_delay=0,
        initial_delay=0,
    )


empty_response = json.loads(request_with_chunks([b""]))
assert "error" in empty_response

oversized_response = json.loads(
    request_with_chunks([b"x" * (helper.MAX_RESPONSE_BYTES + 1)])
)
assert oversized_response == {"error": "VSOCK response exceeded size limit"}

valid_response = '{"ok":true}'
assert request_with_chunks([valid_response.encode(), b""]) == valid_response

print("vsock helper hardening tests passed")
