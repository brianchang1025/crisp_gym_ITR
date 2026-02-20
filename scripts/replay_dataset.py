"""Simple dataset replay utility for LeRobotDataset.

Usage:
    python scripts/replay_dataset.py --repo-id username/dataset_name

The script shows the first available image feature (e.g. observation.images.front)
and prints the action vector and episode index. Use `q` to quit, `n` to advance
one frame when in step mode, or run continuously at the dataset FPS.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rich import print

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobotDataset")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo id of the dataset")
    parser.add_argument("--fps", type=int, default=None, help="Playback fps (defaults to dataset fps)")
    parser.add_argument("--step", action="store_true", help="Step through frames with 'n' key")
    parser.add_argument("--image-key", type=str, default=None, help="Exact image feature key to display")
    args = parser.parse_args()

    ds = LeRobotDataset(repo_id=args.repo_id)
    print(f"Loaded dataset: {args.repo_id}")
    print(f"Total episodes (meta): {ds.meta.total_episodes if hasattr(ds, 'meta') else 'unknown'}")

    # Detect image keys
    image_keys = [k for k in ds.features.keys() if k.startswith("observation.images")]
    if not image_keys:
        print("[red]No image features found in dataset. Available features:[/red]")
        for k in ds.features.keys():
            print(f" - {k}")
        return

    if args.image_key:
        if args.image_key not in image_keys:
            print(f"[red]Requested image key '{args.image_key}' not found. Available: {image_keys}[/red]")
            return
        img_key = args.image_key
    else:
        img_key = image_keys[0]

    print(f"Displaying image feature: {img_key}")

    playback_fps = args.fps if args.fps is not None else getattr(ds, "fps", 15)
    delay_ms = max(1, int(1000.0 / playback_fps))

    last_episode = -1
    window_name = f"replay: {Path(args.repo_id).name} - {img_key}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame in ds:
        ep = int(frame.get("episode_index", 0))
        if ep != last_episode:
            print(f"\n--- Episode {ep} --- task: {frame.get('task','')} \n")
            last_episode = ep

        img = frame.get(img_key)
        if img is None:
            # skip frames without image
            continue

        # Convert C H W -> H W C if needed
        try:
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img_disp = np.moveaxis(img, 0, -1)
            else:
                img_disp = img
        except Exception:
            img_disp = img

        # Normalize or convert types for display
        if img_disp.dtype != np.uint8:
            img_min, img_max = img_disp.min(), img_disp.max()
            if img_max > img_min:
                img_disp = ((img_disp - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            else:
                img_disp = (img_disp * 255.0).astype(np.uint8)

        # Ensure color ordering is BGR for OpenCV (assume images are RGB)
        if img_disp.ndim == 3 and img_disp.shape[2] == 3:
            img_display = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
        else:
            img_display = img_disp

        action = frame.get("action")
        if action is not None:
            action_text = np.array2string(np.asarray(action), precision=3, separator=", ")
        else:
            action_text = "(no action)"

        # Overlay action text on image
        disp = img_display.copy()
        try:
            cv2.putText(disp, f"ep={ep} act={action_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        except Exception:
            pass

        cv2.imshow(window_name, disp)

        if args.step:
            k = cv2.waitKey(0) & 0xFF
        else:
            k = cv2.waitKey(delay_ms) & 0xFF

        if k == ord("q"):
            break
        if args.step and k != ord("n"):
            # wait until 'n' or 'q' is pressed
            if k == ord("q"):
                break
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
