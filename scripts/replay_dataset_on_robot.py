"""Replay actions stored in a LeRobotDataset on a real robot.

Follows the same structure as replay_dataset.py / lerobot_dataset_viz.py.

Example:
    python scripts/replay_dataset_on_robot.py \
        --repo-id username/my_dataset \
        --env-type right_aloha_franka \
        --namespace right

This script:
- loads a LeRobotDataset
- creates a crisp_gym manipulator environment
- visualizes all camera / state / action data via Rerun (same as lerobot_dataset_viz)
- replays the recorded `action` frames on the robot in dataset order

Safety notes:
- Ensure your environment configuration matches the one used for recording.
- Start with low speed / supervised setup and emergency stop available.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import rclpy
import rerun as rr
import torch
import torch.utils.data
import tqdm

import crisp_gym  # noqa: F401
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.util import prompt
from crisp_gym.util.setup_logger import setup_logging
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


logger = logging.getLogger(__name__)


def _to_numpy_1d(value) -> np.ndarray:  # noqa: ANN001
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(-1)


def replay_dataset_on_robot(
    dataset: LeRobotDataset,
    episode_index: int | None = None,
    env_type: str | None = None,
    joint_control: bool = False,
    namespace: str = "right",
    config_path: str | None = None,
    fps: float | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    block: bool = False,
    home_before_start: bool = False,
    home_on_exit: bool = True,
    start_episode: int = 0,
    max_episodes: int | None = None,
    max_frames: int | None = None,
    display_compressed_images: bool = False,
    **kwargs,
) -> Path | None:
    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    run_name = (
        f"{dataset.repo_id}/robot_replay_episode_{episode_index}"
        if episode_index is not None
        else f"{dataset.repo_id}/robot_replay_all_episodes"
    )
    rr.init(run_name, spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=server_uri)

    control_type = "joint" if joint_control else "cartesian"
    logging.info(f"Using control type: {control_type}")

    env = make_env(
        env_type=env_type,
        control_type=control_type,
        namespace=namespace,
        config_path=config_path,
    )

    playback_fps = fps if fps is not None else float(dataset.fps)
    dt = 1.0 / playback_fps if playback_fps > 0 else 0.0

    warned_shape_mismatch = False
    frames_replayed = 0
    episodes_replayed = 0
    current_episode = None
    first_index = None

    try:
        env.wait_until_ready()
        if home_before_start:
            logging.info("Homing robot before replay...")
            env.home()
        env.reset()

        logging.info("Starting replay...")

        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            if first_index is None:
                first_index = batch["index"][0].item()

            for i in range(len(batch["index"])):
                ep_idx = batch["episode_index"][i].item()

                if ep_idx < start_episode:
                    continue

                # Episode boundary: home + reset
                if current_episode is None or ep_idx != current_episode:
                    current_episode = ep_idx
                    episodes_replayed += 1

                    if max_episodes is not None and episodes_replayed > max_episodes:
                        logging.info("Reached --max-episodes limit. Stopping replay.")
                        return None

                    logging.info(f"--- Replaying episode {ep_idx} ---")
                    logging.info("Homing robot at start of episode...")
                    env.home()
                    env.reset()

                # --- Rerun logging (mirrors replay_dataset.py / lerobot_dataset_viz) ---
                rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
                rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

                for key in dataset.meta.camera_keys:
                    img = to_hwc_uint8_numpy(batch[key][i])
                    img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                    rr.log(key, entity=img_entity)

                if ACTION in batch:
                    for dim_idx, val in enumerate(batch[ACTION][i]):
                        rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

                if OBS_STATE in batch:
                    for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                        rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

                if DONE in batch:
                    rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

                if REWARD in batch:
                    rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

                if "next.success" in batch:
                    rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

                if "task" in batch:
                    rr.log("task", rr.TextLog(str(batch["task"][i])))

                # --- Robot action ---
                if ACTION not in batch:
                    continue

                action_np = _to_numpy_1d(batch[ACTION][i])
                expected_dim = int(env.action_space.shape[0])

                if action_np.shape[0] < expected_dim:
                    raise ValueError(
                        f"Action dim {action_np.shape[0]} is smaller than env action dim {expected_dim}. "
                        "Please use the same env/control config used during data recording."
                    )
                if action_np.shape[0] > expected_dim:
                    if not warned_shape_mismatch:
                        logging.warning(
                            f"Action dim {action_np.shape[0]} > env action dim {expected_dim}. "
                            "Truncating extra dimensions."
                        )
                        warned_shape_mismatch = True
                    action_np = action_np[:expected_dim]

                env.step(action_np.astype(np.float32), block=block)

                frames_replayed += 1
                if max_frames is not None and frames_replayed >= max_frames:
                    logging.info("Reached --max-frames limit. Stopping replay.")
                    return None

                if not block and dt > 0:
                    time.sleep(dt)

        logging.info(
            f"Replay finished. Episodes: {episodes_replayed}, frames: {frames_replayed}"
        )

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    finally:
        if home_on_exit:
            try:
                logging.info("Homing robot on exit...")
                env.home()
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Failed to home on exit: {e}")

        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Failed to close environment cleanly: {e}")

        if rclpy.ok():
            rclpy.shutdown()

    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = dataset.repo_id.replace("/", "_")
        suffix = f"episode_{episode_index}" if episode_index is not None else "all_episodes"
        rrd_path = output_dir / f"{repo_id_str}_{suffix}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")

    return None


def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobotDataset on robot hardware")

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        "--episode",
        type=int,
        default=None,
        help="Episode to replay. If not set, defaults to episode 0.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally. By default, uses HF cache.",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default=None,
        help="Environment config name (e.g. right_aloha_franka).",
    )
    parser.add_argument(
        "--joint-control",
        action="store_true",
        help="Use joint control instead of cartesian control.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Robot namespace in ROS2 (e.g. left, right).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional explicit path to env YAML config.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override (defaults to dataset fps).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' spawns a viewer locally. "
            "'distant' creates a gRPC server; connect with `rerun rerun+http://IP:GRPC_PORT/proxy`."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "Also deactivates spawning of a local viewer."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Tolerance in seconds used to ensure data timestamps respect the dataset fps value.",
    )
    parser.add_argument(
        "--block",
        action="store_true",
        help="Use blocking env.step (lets env control loop handle timing).",
    )
    parser.add_argument(
        "--home-before-start",
        action="store_true",
        help="Home robot before replay begins.",
    )
    parser.add_argument(
        "--home-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Home robot when script exits.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index to start from.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Replay at most this many episodes.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Replay at most this many frames.",
    )
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="Display compressed images in Rerun (same behavior as lerobot_dataset_viz).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()
    kwargs = vars(args)

    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    log_level = kwargs.pop("log_level")

    setup_logging(level=getattr(logging, log_level.upper(), logging.INFO))

    # --- Interactive prompts (when args not provided) ---

    if repo_id is None:
        repo_id_input = input(
            "Please enter repo id (e.g. cbrian/dataset_picktheredscrew_cartesian): "
        ).strip()
        if not repo_id_input:
            raise ValueError("repo-id is required")
        repo_id = repo_id_input

    if args.episode_index is None:
        ep_input = input("Episode index (default: 0, type 'all' for all episodes): ").strip().lower()
        if ep_input == "" or ep_input == "0":
            args.episode_index = 0
        elif ep_input == "all":
            args.episode_index = None
        else:
            args.episode_index = int(ep_input)

    if args.env_type is None:
        env_configs = list_env_configs()
        args.env_type = prompt.prompt(
            "Please select the follower robot environment configuration.",
            options=env_configs,
            default=env_configs[0] if env_configs else None,
        )

    if args.namespace is None:
        args.namespace = prompt.prompt(
            "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
            default="right",
        )

    logging.info("Loading dataset")
    if args.episode_index is None:
        dataset = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
    else:
        dataset = LeRobotDataset(
            repo_id,
            episodes=[args.episode_index],
            root=root,
            tolerance_s=tolerance_s,
        )

    replay_dataset_on_robot(dataset, **vars(args))


if __name__ == "__main__":
    main()
