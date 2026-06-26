"""Laptop-side rollout loop for network-split ACT inference.

Connects to robot_server.py on the Pi, loads the ACT policy, and runs a
chunked inference loop:
  - One policy.predict_action_chunk() call per 100 control ticks
  - Between chunk calls: return next buffered action (sub-ms per tick)
  - Effective stall: ~1 inference per ~4.2s of execution at 24 Hz

Usage (from laptop, with LeRobot env active):
  python rollout_remote.py --robot-host <pi-ip> --policy /path/to/act_so101 --task "pick and place"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from network_robot import MOTOR_KEYS, NetworkRobot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_ds_features(policy_path: str) -> dict:
    """Load dataset features from the policy checkpoint directory."""
    meta_path = Path(policy_path) / "meta" / "info.json"
    if not meta_path.exists():
        meta_path = Path(policy_path) / "dataset_meta" / "info.json"
    if meta_path.exists():
        with open(meta_path) as f:
            info = json.load(f)
        return info["features"]

    # Fallback: load from the cached dataset metadata
    dataset_meta = Path.home() / ".cache/huggingface/lerobot/l-e-n/so101_pick_place_50-merged/meta/info.json"
    if dataset_meta.exists():
        with open(dataset_meta) as f:
            info = json.load(f)
        log.info("Loaded ds_features from dataset cache at %s", dataset_meta)
        return info["features"]

    raise FileNotFoundError(
        f"Could not find info.json in {policy_path} or dataset cache. "
        "Pass --dataset-meta /path/to/meta/info.json explicitly."
    )


def run_rollout(
    robot_host: str,
    robot_port: int,
    policy_path: str,
    task: str,
    duration: float,
    device_str: str,
) -> None:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.utils import build_inference_frame
    from lerobot.utils.feature_utils import build_dataset_frame

    device = torch.device(device_str)

    log.info("Loading policy from %s ...", policy_path)
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.to(device)

    ds_features = _load_ds_features(policy_path)

    log.info("Connecting to Pi at %s:%d ...", robot_host, robot_port)
    robot = NetworkRobot(robot_host, robot_port)
    robot.connect()
    log.info("Connected")

    action_chunk: torch.Tensor | None = None  # shape [1, chunk_size, 6] when loaded
    chunk_idx = 0

    inference_times: list[float] = []
    tick_times: list[float] = []
    start_time = time.perf_counter()
    tick = 0

    try:
        while True:
            if duration > 0 and (time.perf_counter() - start_time) >= duration:
                log.info("Duration %.0fs reached, stopping", duration)
                break

            t_tick = time.perf_counter()
            obs = robot.get_observation()

            obs_frame = build_inference_frame(
                obs,
                device=device,
                ds_features=ds_features,
                task=task,
                robot_type="so101_follower",
            )

            if action_chunk is None or chunk_idx >= action_chunk.shape[1]:
                t_inf = time.perf_counter()
                with torch.inference_mode():
                    action_chunk = policy.predict_action_chunk(obs_frame)  # [1, chunk_size, 6]
                inf_time = time.perf_counter() - t_inf
                inference_times.append(inf_time)
                chunk_idx = 0
                log.info("inference %.3fs  chunk_size=%d", inf_time, action_chunk.shape[1])

            action_tensor = action_chunk[0, chunk_idx].cpu()  # [6]
            chunk_idx += 1

            action_dict = {k: action_tensor[i].item() for i, k in enumerate(MOTOR_KEYS)}
            robot.send_action(action_dict)

            tick_times.append(time.perf_counter() - t_tick)
            tick += 1

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        robot.disconnect()

    _print_summary(tick, tick_times, inference_times)


def _print_summary(tick: int, tick_times: list[float], inference_times: list[float]) -> None:
    if not tick_times:
        return
    mean_tick_ms = sum(tick_times) / len(tick_times) * 1000
    effective_hz = 1.0 / (sum(tick_times) / len(tick_times))
    print(f"\n--- Rollout summary ---")
    print(f"  Total ticks      : {tick}")
    print(f"  Mean tick time   : {mean_tick_ms:.1f}ms")
    print(f"  Effective Hz     : {effective_hz:.2f}")
    if inference_times:
        mean_inf = sum(inference_times) / len(inference_times)
        print(f"  Inference calls  : {len(inference_times)}")
        print(f"  Mean inference   : {mean_inf:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop-side rollout for network-split ACT inference")
    parser.add_argument("--robot-host", required=True, help="Pi IP address")
    parser.add_argument("--robot-port", type=int, default=9000, help="Pi TCP port (default: 9000)")
    parser.add_argument(
        "--policy",
        default="/home/len/checkpoints/act_so101",
        help="Path to policy checkpoint directory",
    )
    parser.add_argument("--task", default="pick and place", help="Task description string")
    parser.add_argument("--duration", type=float, default=60.0, help="Run duration in seconds (0=infinite)")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu, cuda, mps)")
    args = parser.parse_args()

    run_rollout(
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        policy_path=args.policy,
        task=args.task,
        duration=args.duration,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
