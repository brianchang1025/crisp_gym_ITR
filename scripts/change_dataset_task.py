#!/usr/bin/env python3
"""Change the task description for all episodes in a LeRobot dataset.

Usage: python scripts/change_dataset_task.py --repo-id SOURCE/REPO --output-repo-id OUT/REPO --task "new task"
"""
from pathlib import Path
import shutil
import argparse
import sys
import tempfile

from huggingface_hub import snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import load_episodes, write_episodes, write_tasks, load_info, write_json
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Replace task description for all episodes in a dataset")
    p.add_argument("--repo-id", required=True, help="Source dataset repo id (e.g. cbrian/my_dataset)")
    p.add_argument("--output-repo-id", required=True, help="Output local repo id (will be created under HF_LEROBOT_HOME)")
    p.add_argument("--task", required=True, help="New task description to set for every episode")
    p.add_argument("--force-overwrite", action="store_true", help="Overwrite output path if it exists locally")
    p.add_argument("--push-to-hub", action="store_true", help="Push the resulting dataset to the Hugging Face Hub")
    p.add_argument("--private", action="store_true", help="Create the repo as private when pushing to the Hub")
    p.add_argument("--push-videos", action="store_true", help="Also push the videos/ directory to the Hub")
    p.add_argument("--branch", default=None, help="Branch/revision to push to on the Hub")
    p.add_argument("--tags", default=None, help="Comma-separated tags to add to the dataset card")
    args = p.parse_args()

    src_repo = args.repo_id
    out_repo = args.output_repo_id
    new_task = args.task

    # Load source metadata (this will download metadata files if needed)
    print(f"Loading metadata for {src_repo}...")
    meta = LeRobotDatasetMetadata(src_repo, force_cache_sync=True)

    # Ensure episodes are loaded
    episodes = meta.episodes if getattr(meta, "episodes", None) is not None else load_episodes(meta.root)

    try:
        import datasets
    except Exception:
        print("Missing dependency 'datasets'. Please install 'datasets' package.", file=sys.stderr)
        sys.exit(2)

    if not isinstance(episodes, datasets.Dataset):
        print("Could not load episodes dataset", file=sys.stderr)
        sys.exit(2)

    # Replace tasks for every episode
    print(f"Setting task for {len(episodes)} episodes to: {new_task}")

    def _set_task(example):
        example["tasks"] = [new_task]
        return example

    episodes_mod = episodes.map(_set_task)

    # Prepare output local directory
    out_root = meta.root.parent / out_repo
    if out_root.exists():
        if args.force_overwrite:
            shutil.rmtree(out_root)
        else:
            print(f"Output path {out_root} already exists. Use --force-overwrite to replace.")
            sys.exit(1)

    # The metadata object may have been downloaded with only meta files.
    # Download the full repository (data + videos) to a temporary folder and copy from there.
    print(f"Downloading full dataset {src_repo} to temporary folder...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="lr_ds_"))
    snapshot_download(repo_id=src_repo, repo_type="dataset", local_dir=str(tmp_dir))
    # snapshot_download places files under tmp_dir / <repo_id-basename>
    src_root = tmp_dir / Path(src_repo).name
    if not src_root.exists():
        # fallback: sometimes snapshot_download writes directly into tmp_dir
        src_root = tmp_dir

    print(f"Copying dataset files to {out_root}... (this may take a while for large datasets)")
    shutil.copytree(src_root, out_root)

    # Write modified episodes metadata
    print("Writing modified episodes metadata...")
    write_episodes(episodes_mod, out_root)

    # Write tasks parquet (single task)
    print("Updating tasks list to single new task...")
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[new_task])
    write_tasks(tasks_df, out_root)

    # Update info.json total_tasks
    info = load_info(out_root)
    info["total_tasks"] = 1
    write_json(info, Path(out_root) / "meta" / "info.json")

    print("Done. Local modified dataset is at:", out_root)

    if args.push_to_hub:
        print(f"Pushing {out_repo} to the Hugging Face Hub (this may take a while)...")
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        # Instantiate LeRobotDataset with the local root so push_to_hub uploads files correctly
        ds = LeRobotDataset(out_repo, root=out_root)
        ds.push_to_hub(branch=args.branch, tags=tags, private=args.private, push_videos=args.push_videos)
        print("Push complete.")


if __name__ == "__main__":
    main()
