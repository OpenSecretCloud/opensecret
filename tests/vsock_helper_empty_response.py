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

SUCCESSFUL_RESPONSE_BYTES = b'{"ok": true, "message": "unchanged"}'


def expected_error(max_retries):
    return json.dumps(
        {"error": f"VSOCK connection failed after {max_retries} attempts"}
    )


class SuccessfulResponseSocket:
    def __init__(self):
        # Exercise the existing multi-chunk path and retain the exact payload.
        self.chunks = [
            SUCCESSFUL_RESPONSE_BYTES[:7],
            SUCCESSFUL_RESPONSE_BYTES[7:19],
            SUCCESSFUL_RESPONSE_BYTES[19:],
            b"",
        ]

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


class EmptyResponseSocket:
    def close(self):
        pass

    def connect(self, _address):
        pass

    def recv(self, _length):
        return b""

    def sendall(self, _request):
        pass

    def settimeout(self, _seconds):
        pass


class ReceiveFailureSocket(EmptyResponseSocket):
    def __init__(self, error):
        self.error = error

    def recv(self, _length):
        raise self.error


successful_attempts = 0


def successful_response_socket(*_args, **_kwargs):
    global successful_attempts
    successful_attempts += 1
    return SuccessfulResponseSocket()


helper.socket.socket = successful_response_socket
helper.select.select = lambda *_args, **_kwargs: ([object()], [], [])
helper.time.sleep = lambda _seconds: None

successful_response = helper.vsock_request(
    3,
    8003,
    "{}",
    max_retries=3,
    retry_delay=0,
    initial_delay=0,
)

assert successful_attempts == 1
assert successful_response.encode() == SUCCESSFUL_RESPONSE_BYTES
assert json.loads(successful_response) == {"ok": True, "message": "unchanged"}


retry_then_success_attempts = 0


def empty_then_successful_response_socket(*_args, **_kwargs):
    global retry_then_success_attempts
    retry_then_success_attempts += 1
    if retry_then_success_attempts == 1:
        return EmptyResponseSocket()
    return SuccessfulResponseSocket()


helper.socket.socket = empty_then_successful_response_socket

retry_then_success_response = helper.vsock_request(
    3,
    8003,
    "{}",
    max_retries=3,
    retry_delay=0,
    initial_delay=0,
)

assert retry_then_success_attempts == 2
assert retry_then_success_response.encode() == SUCCESSFUL_RESPONSE_BYTES
assert json.loads(retry_then_success_response) == {
    "ok": True,
    "message": "unchanged",
}


attempts = 0


def empty_response_socket(*_args, **_kwargs):
    global attempts
    attempts += 1
    return EmptyResponseSocket()


helper.socket.socket = empty_response_socket
helper.select.select = lambda *_args, **_kwargs: ([object()], [], [])
helper.time.sleep = lambda _seconds: None

max_retries = 3
response = helper.vsock_request(
    3,
    8003,
    "{}",
    max_retries=max_retries,
    retry_delay=0,
    initial_delay=0,
)

assert attempts == max_retries
assert response == expected_error(max_retries)


def assert_receive_failure_fails_closed(error):
    receive_attempts = 0

    def receive_failure_socket(*_args, **_kwargs):
        nonlocal receive_attempts
        receive_attempts += 1
        return ReceiveFailureSocket(error)

    helper.socket.socket = receive_failure_socket
    helper.select.select = lambda *_args, **_kwargs: ([object()], [], [])

    response = helper.vsock_request(
        3,
        8003,
        "{}",
        max_retries=max_retries,
        retry_delay=0,
        initial_delay=0,
    )

    assert receive_attempts == max_retries
    assert response == expected_error(max_retries)


assert_receive_failure_fails_closed(helper.socket.error("fixture read error"))
assert_receive_failure_fails_closed(helper.socket.timeout("fixture read timeout"))


def unexpected_socket(*_args, **_kwargs):
    raise AssertionError("zero retries must not create a socket")


helper.socket.socket = unexpected_socket
zero_retry_response = helper.vsock_request(
    3,
    8003,
    "{}",
    max_retries=0,
    retry_delay=0,
    initial_delay=0,
)
assert zero_retry_response == expected_error(0)

print("vsock helper empty-response regression test passed")
