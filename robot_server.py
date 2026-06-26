#!/usr/bin/env python3
"""TCP server that runs on the Pi, owning hardware I/O for network-split inference.

The laptop connects, then the server loops:
  1. Read observation from the robot (cameras + motor state)
  2. Serialize and send to laptop
  3. Receive action from laptop
  4. Write action to robot motors
  5. Sleep to maintain target FPS

Wire format (both directions): 4-byte LE length prefix + msgpack payload.

Pi → Laptop observation packet:
  {"motor_pos": [float*6], "wrist_img": bytes, "front_img": bytes, "timestamp": float}

Laptop → Pi action packet:
  {"motor_pos": [float*6]}

Motor key order (must match laptop side):
  shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
  wrist_flex.pos, wrist_roll.pos, gripper.pos
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import time

import msgpack
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

WRIST_CAM_KEY = "wrist"
FRONT_CAM_KEY = "front"


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------

def _send_msg(sock: socket.socket, data: bytes) -> None:
    header = struct.pack("<I", len(data))
    sock.sendall(header + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by remote")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 4)
    length = struct.unpack("<I", header)[0]
    return _recv_exact(sock, length)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _obs_to_packet(obs: dict) -> dict:
    motor_pos = [float(obs[k]) for k in MOTOR_KEYS]
    wrist_img: np.ndarray = obs[WRIST_CAM_KEY]
    front_img: np.ndarray = obs[FRONT_CAM_KEY]
    return {
        "motor_pos": motor_pos,
        "wrist_img": wrist_img.tobytes(),
        "front_img": front_img.tobytes(),
        "wrist_shape": list(wrist_img.shape),
        "front_shape": list(front_img.shape),
        "timestamp": time.time(),
    }


def _packet_to_action(packet: dict) -> dict:
    return {k: float(v) for k, v in zip(MOTOR_KEYS, packet["motor_pos"])}


# ---------------------------------------------------------------------------
# Robot server
# ---------------------------------------------------------------------------

def _build_robot(robot_port: str):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(
        port=robot_port,
        use_degrees=True,
        cameras={
            WRIST_CAM_KEY: OpenCVCameraConfig(
                index_or_path="/dev/video0",
                fps=30,
                width=640,
                height=480,
            ),
            FRONT_CAM_KEY: OpenCVCameraConfig(
                index_or_path="/dev/video2",
                fps=30,
                width=640,
                height=480,
            ),
        },
    )
    return SO101Follower(config)


def run_server(host: str, port: int, fps: float, robot_port: str) -> None:
    control_interval = 1.0 / fps

    log.info("Connecting robot on %s...", robot_port)
    robot = _build_robot(robot_port)
    robot.connect(calibrate=False)
    log.info("Robot connected")

    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind((host, port))
    srv_sock.listen(1)
    log.info("Listening on %s:%d", host, port)

    try:
        while True:
            log.info("Waiting for laptop connection...")
            conn, addr = srv_sock.accept()
            log.info("Connected: %s", addr)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                _serve_connection(conn, robot, control_interval)
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                log.warning("Connection lost: %s", e)
            finally:
                conn.close()
    finally:
        robot.disconnect()
        srv_sock.close()


def _serve_connection(
    conn: socket.socket, robot, control_interval: float
) -> None:
    tick = 0
    while True:
        t0 = time.perf_counter()

        obs = robot.get_observation()

        packet = _obs_to_packet(obs)
        _send_msg(conn, msgpack.packb(packet, use_bin_type=True))

        raw = _recv_msg(conn)
        action_packet = msgpack.unpackb(raw, raw=False)
        action_dict = _packet_to_action(action_packet)
        robot.send_action(action_dict)

        dt = time.perf_counter() - t0
        tick += 1
        if tick % 24 == 0:
            log.info("tick=%d  cycle=%.1fms", tick, dt * 1000)

        sleep_t = control_interval - dt
        if sleep_t > 0:
            time.sleep(sleep_t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pi-side robot server for network split inference")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="TCP port (default: 9000)")
    parser.add_argument("--fps", type=float, default=24.0, help="Target control rate (default: 24)")
    parser.add_argument(
        "--robot-port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14115684-if00",
        help="Serial port for SO-101 follower arm",
    )
    args = parser.parse_args()
    run_server(args.host, args.port, args.fps, args.robot_port)


if __name__ == "__main__":
    main()
