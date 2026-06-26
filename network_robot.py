"""Client-side robot interface for network-split inference.

Runs on the laptop. Wraps the TCP connection to the Pi server and presents
get_observation() / send_action() so rollout_remote.py reads like a normal
robot loop.

The Pi server drives the loop — it sends the first observation immediately
after connection without waiting for any action.
"""

from __future__ import annotations

import socket
import struct
import time

import msgpack
import numpy as np

MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _send_msg(sock: socket.socket, data: bytes) -> None:
    header = struct.pack("<I", len(data))
    sock.sendall(header + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by Pi server")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 4)
    length = struct.unpack("<I", header)[0]
    return _recv_exact(sock, length)


class NetworkRobot:
    """Thin TCP client that proxies robot I/O from the Pi."""

    def __init__(self, host: str, port: int = 9000, timeout: float = 10.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect((self._host, self._port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def disconnect(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def get_observation(self) -> dict:
        """Receive the next observation from the Pi server.

        Returns a dict with:
          - Individual motor keys: "shoulder_pan.pos", ..., "gripper.pos" (float)
          - "wrist": np.ndarray(H, W, 3) uint8
          - "front": np.ndarray(H, W, 3) uint8
          - "timestamp": float

        This dict matches the shape of SO101Follower.get_observation() so
        LeRobot utilities (build_inference_frame, etc.) work without modification.
        """
        raw = _recv_msg(self._sock)
        packet = msgpack.unpackb(raw, raw=False)

        obs = {k: packet["motor_pos"][i] for i, k in enumerate(MOTOR_KEYS)}

        wrist_shape = tuple(packet["wrist_shape"])
        front_shape = tuple(packet["front_shape"])
        obs["wrist"] = np.frombuffer(packet["wrist_img"], dtype=np.uint8).reshape(wrist_shape)
        obs["front"] = np.frombuffer(packet["front_img"], dtype=np.uint8).reshape(front_shape)
        obs["timestamp"] = packet["timestamp"]

        return obs

    def send_action(self, action_dict: dict) -> None:
        """Send a motor position command to the Pi server."""
        motor_pos = [float(action_dict[k]) for k in MOTOR_KEYS]
        payload = msgpack.packb({"motor_pos": motor_pos}, use_bin_type=True)
        _send_msg(self._sock, payload)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
