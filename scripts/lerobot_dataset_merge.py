#!/usr/bin/env python3
"""Merge multiple LeRobot datasets and upload to HuggingFace.

Usage:
    pixi run -e jazzy-lerobot python scripts/lerobot_dataset+check.py \
        --datasets cbrian/dataset_A cbrian/dataset_B cbrian/dataset_C \
        --output cbrian/dataset_merged \
        [--local-dir ./merged_output] \
        [--public]
"""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Keys managed internally by the dataset — do NOT pass to add_frame
INTERNAL_KEYS = {"frame_index", "episode_index", "index", "task_index", "timestamp"}


def normalize_frame_for_add(frame: dict, video_keys: set[str]) -> dict:
    """Convert loaded frame into the format expected by LeRobotDataset.add_frame()."""
    clean = {k: v for k, v in frame.items() if k not in INTERNAL_KEYS}

    for key, value in list(clean.items()):
        arr = value
        if hasattr(arr, "detach") and hasattr(arr, "cpu"):
            arr = arr.detach().cpu().numpy()

        if key in video_keys:
            # ds[i] returns CHW torch tensors; add_frame expects HWC arrays
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
        elif key == "observation.state.gripper":
            # Some datasets expose this as a scalar; feature expects shape (1,)
            if np.isscalar(arr) or (isinstance(arr, np.ndarray) and arr.ndim == 0):
                arr = np.asarray([arr], dtype=np.float32)

        clean[key] = arr

    return clean


def load_datasets(repo_ids: list) -> list:
    datasets = []
    for repo_id in repo_ids:
        print(f"Loading: {repo_id}")
        ds = LeRobotDataset(repo_id)
        print(f"  → {ds.num_episodes} episodes, {ds.num_frames} frames, {ds.fps} fps")
        tasks_df = ds.meta.tasks
        if "task" in tasks_df.columns:
            task_names = list(tasks_df["task"])
        else:
            task_names = list(tasks_df.index)
        print(f"  → tasks: {task_names}")
        datasets.append(ds)
    return datasets


def check_compatibility(datasets: list):
    """Ensure all datasets share the same fps and feature set."""
    ref = datasets[0]
    for ds in datasets[1:]:
        if ds.fps != ref.fps:
            raise ValueError(
                f"FPS mismatch: {ref.meta.repo_id} ({ref.fps} fps) vs "
                f"{ds.meta.repo_id} ({ds.fps} fps)"
            )
        diff = set(ds.features.keys()).symmetric_difference(set(ref.features.keys()))
        if diff:
            raise ValueError(
                f"Feature mismatch between {ref.meta.repo_id} and "
                f"{ds.meta.repo_id}: differing keys = {diff}"
            )
    print(f"\n✓ All {len(datasets)} datasets are compatible\n")


def iter_episode_frames(ds: LeRobotDataset, episode_idx: int):
    """Yield all frame dicts for a given episode index."""
    ep_meta = ds.meta.episodes[episode_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    for i in range(ep_from, ep_to):
        yield ds[i]


def merge_and_upload(source_datasets: list, output_repo_id: str,
                     local_dir: str, private: bool):
    ref = source_datasets[0]
    root = Path(local_dir) if local_dir else None

    total_episodes = sum(ds.num_episodes for ds in source_datasets)
    total_frames = sum(ds.num_frames for ds in source_datasets)
    print(f"Creating output dataset: {output_repo_id}")
    print(f"Total episodes to merge: {total_episodes}, frames: {total_frames}\n")

    out_ds = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=ref.fps,
        features=ref.features,
        root=root,
        robot_type=ref.meta.robot_type,
        use_videos=bool(ref.meta.video_keys),
    )
    video_keys = set(ref.meta.video_keys)

    episode_count = 0
    for src_ds in source_datasets:
        print(f"Merging {src_ds.meta.repo_id}  ({src_ds.num_episodes} episodes)")
        for ep_idx in range(src_ds.num_episodes):
            for frame in iter_episode_frames(src_ds, ep_idx):
                clean = normalize_frame_for_add(frame, video_keys)
                out_ds.add_frame(clean)
            out_ds.save_episode()
            episode_count += 1
            print(f"  [{episode_count}/{total_episodes}] episode {ep_idx} saved", end="\r")
        print()  # newline after each source dataset

    print("\nFinalizing dataset...")
    out_ds.finalize()

    print(f"Pushing to HuggingFace as {'private' if private else 'public'}: {output_repo_id}")
    out_ds.push_to_hub(private=private)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets and upload to HuggingFace"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="One or more source dataset repo IDs",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output HuggingFace repo ID for the merged dataset",
    )
    parser.add_argument(
        "--local-dir", default=None,
        help="Local directory to write the merged dataset (default: HF cache)",
    )
    parser.add_argument(
        "--public", dest="private", action="store_false", default=True,
        help="Upload as a public repo (default: private)",
    )
    args = parser.parse_args()

    datasets = load_datasets(args.datasets)
    check_compatibility(datasets)
    merge_and_upload(datasets, args.output, args.local_dir, args.private)


if __name__ == "__main__":
    main()
