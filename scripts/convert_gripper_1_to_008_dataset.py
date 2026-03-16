#!/usr/bin/env python3
"""Clone a LeRobotDataset while remapping gripper values.

This script loads all frames from a source LeRobot dataset, replaces
`action.gripper == 1.0` with `0.08`, and writes the full result
as a new LeRobot dataset.

Example:
    pixi run -e jazzy-lerobot python scripts/convert_gripper_1_to_008_dataset.py \
        --src-repo-id cbrian/dataset_picktheredscrew_cartesian \
        --dst-repo-id cbrian/dataset_picktheredscrew_cartesian_gripper008
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


AUTO_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _replace_gripper_value(x, from_val: float = 1.0, to_val: float = 0.08, atol: float = 1e-6):
    """Replace values equal (within tolerance) to `from_val` with `to_val`."""
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy().copy()
        mask = np.isclose(arr, from_val, atol=atol)
        arr[mask] = to_val
        return arr.astype(np.float32)

    arr = np.asarray(x).copy()
    mask = np.isclose(arr, from_val, atol=atol)
    arr[mask] = to_val
    return arr.astype(np.float32)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _coerce_to_expected_shape(value, expected_shape: tuple[int, ...]):
    arr = _to_numpy(value)

    # Common image case: source frame provides CHW, dataset expects HWC.
    if arr.ndim == 3 and len(expected_shape) == 3 and tuple(arr.shape) != expected_shape:
        if tuple(arr.shape) == (expected_shape[2], expected_shape[0], expected_shape[1]):
            arr = np.transpose(arr, (1, 2, 0))

    # Common scalar->vector case (e.g. gripper expected shape (1,)).
    if len(expected_shape) == 1 and expected_shape[0] == 1 and arr.shape == ():
        arr = arr.reshape(1)

    # Last resort: if same number of elements, reshape to expected.
    if arr.size == int(np.prod(expected_shape)) and tuple(arr.shape) != expected_shape:
        arr = arr.reshape(expected_shape)

    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a LeRobot dataset, replace action.gripper values 1.0 -> 0.08, "
            "and save as a new dataset."
        )
    )
    parser.add_argument("--src-repo-id", required=True, help="Source dataset repo id")
    parser.add_argument("--dst-repo-id", required=True, help="Destination dataset repo id")
    parser.add_argument("--src-root", default=None, help="Optional source root directory")
    parser.add_argument("--dst-root", default=None, help="Optional destination root directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete destination dataset directory if it already exists",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="action.gripper",
        help="Feature key to modify (default: action.gripper).",
    )
    parser.add_argument(
        "--action-index",
        type=int,
        default=-1,
        help="Index inside action vector to modify when target key is 'action' (default: -1).",
    )
    parser.add_argument("--from-value", type=float, default=1.0, help="Value to replace")
    parser.add_argument("--to-value", type=float, default=0.08, help="Replacement value")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for equality check")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="If set, push the converted dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether pushed dataset should be private (used only with --push-to-hub).",
    )
    args = parser.parse_args()

    src = LeRobotDataset(repo_id=args.src_repo_id, root=args.src_root)

    dst_base = Path(args.dst_root) if args.dst_root is not None else HF_LEROBOT_HOME
    dst_path = dst_base / args.dst_repo_id

    if dst_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_path)

    # LeRobotDataset.create expects `root` to be the dataset directory itself.
    # If dst_root is omitted, pass root=None so LeRobot builds HF_LEROBOT_HOME/repo_id.
    dst_dataset_root = dst_path if args.dst_root is not None else None

    # Keep non-auto features; LeRobot will add default index/timestamp/task_index fields itself.
    features_to_copy = {k: v for k, v in src.features.items() if k not in AUTO_FEATURE_KEYS}

    target_key = args.target_key
    action_index = args.action_index

    # Common case: dataset stores only vector `action`, not nested `action.gripper`.
    if target_key == "action.gripper" and "action.gripper" not in features_to_copy and "action" in features_to_copy:
        target_key = "action"
        action_index = -1

    if target_key not in features_to_copy:
        available = sorted(features_to_copy.keys())
        raise KeyError(
            f"Target key '{target_key}' not found in dataset features. "
            f"Available keys: {available}"
        )

    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        root=dst_dataset_root,
        fps=src.fps,
        robot_type=src.meta.robot_type,
        features=features_to_copy,
        use_videos=len(src.meta.video_keys) > 0,
    )

    current_episode = None
    modified_count = 0

    for frame in tqdm(src, total=len(src), desc="Converting frames"):
        ep_idx = _as_int(frame.get("episode_index"), default=0)

        if current_episode is None:
            current_episode = ep_idx
        elif ep_idx != current_episode:
            dst.save_episode()
            current_episode = ep_idx

        out_frame: dict = {}
        for key in features_to_copy:
            if key in frame:
                expected_shape = tuple(features_to_copy[key]["shape"])
                out_frame[key] = _coerce_to_expected_shape(frame[key], expected_shape)

        # Preserve task if present in source frame.
        if "task" in frame:
            out_frame["task"] = str(frame["task"])

        if target_key in out_frame:
            # If target is vector `action`, only change selected element, e.g. action[-1].
            if target_key == "action":
                arr = _to_numpy(out_frame[target_key]).astype(np.float32, copy=True)
                idx = action_index if action_index >= 0 else arr.shape[0] + action_index
                if idx < 0 or idx >= arr.shape[0]:
                    raise IndexError(
                        f"--action-index {action_index} out of bounds for action shape {arr.shape}."
                    )

                before_matches = int(np.isclose(arr[idx], args.from_value, atol=args.atol))
                if before_matches:
                    arr[idx] = np.float32(args.to_value)
                out_frame[target_key] = arr
                modified_count += before_matches
            else:
                before = np.asarray(out_frame[target_key])
                before_matches = int(np.isclose(before, args.from_value, atol=args.atol).sum())

                out_frame[target_key] = _replace_gripper_value(
                    out_frame[target_key],
                    from_val=args.from_value,
                    to_val=args.to_value,
                    atol=args.atol,
                )
                modified_count += before_matches

        dst.add_frame(out_frame)

    # Save final episode if buffer has frames.
    if dst.episode_buffer is not None and dst.episode_buffer.get("size", 0) > 0:
        dst.save_episode()

    dst.finalize()

    if args.push_to_hub:
        print(f"Pushing {args.dst_repo_id} to Hugging Face Hub (private={args.private})...")
        dst.push_to_hub(private=args.private)
        print("Push complete.")

    print("Done.")
    print(f"Source dataset: {args.src_repo_id}")
    print(f"New dataset:    {args.dst_repo_id}")
    print(f"Destination:    {dst_path}")
    print(f"Target key:     {target_key}")
    if target_key == "action":
        print(f"Action index:   {action_index}")
    print(f"Gripper values replaced: {modified_count}")


if __name__ == "__main__":
    main()
