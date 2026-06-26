"""Standalone latency/correctness test for the Pi-laptop TCP connection.

Runs on the laptop (no inference). Connects to robot_server.py, receives N
observations, sends zero-filled actions, and reports timing and data shape.

Usage:
  python test_connection.py --robot-host <pi-ip> --n 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from network_robot import MOTOR_KEYS, NetworkRobot


def run_test(host: str, port: int, n: int) -> None:
    print(f"Connecting to {host}:{port} ...")
    robot = NetworkRobot(host, port)
    robot.connect()
    print("Connected\n")

    rtt_times: list[float] = []

    for i in range(n):
        t0 = time.perf_counter()
        obs = robot.get_observation()

        zero_action = {k: 0.0 for k in MOTOR_KEYS}
        robot.send_action(zero_action)

        rtt = time.perf_counter() - t0
        rtt_times.append(rtt)

        if i == 0:
            # Print data shapes and sample values on first frame
            print("Observation keys and shapes:")
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"  {k}: {v}")
            print()

        if (i + 1) % 5 == 0:
            mean_rtt_ms = sum(rtt_times[-5:]) / 5 * 1000
            print(f"  tick {i+1:3d}/{n}  RTT={mean_rtt_ms:.1f}ms")

    robot.disconnect()

    mean_rtt = sum(rtt_times) / len(rtt_times)
    max_rtt = max(rtt_times)
    print(f"\n--- Summary ---")
    print(f"  Ticks        : {n}")
    print(f"  Mean RTT     : {mean_rtt*1000:.1f}ms")
    print(f"  Max RTT      : {max_rtt*1000:.1f}ms")
    print(f"  Effective Hz : {1/mean_rtt:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency and correctness test for Pi-laptop connection")
    parser.add_argument("--robot-host", required=True, help="Pi IP address")
    parser.add_argument("--robot-port", type=int, default=9000)
    parser.add_argument("--n", type=int, default=20, help="Number of observation/action cycles")
    args = parser.parse_args()
    run_test(args.robot_host, args.robot_port, args.n)


if __name__ == "__main__":
    main()
