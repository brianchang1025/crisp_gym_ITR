"""Interface for Pi05 Policy inference in CRISP."""

import os
# # hide GPU from both main and spawned processes; prevents CUDA allocations
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import gc
import logging
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from transformers import AutoTokenizer
from peft import PeftConfig, PeftModel
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import multiprocessing as mp
import torch
import torch._inductor.config as inductor_config
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import LeRobotDatasetMetadata, get_policy_class
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
inductor_config.triton.cudagraphs = False


@register_policy("pi05_libero_policy")
class Pi05LiberoPolicy(Policy):
    """A policy implementation for Pi05 that wraps a LeRobot policy for edge device inference.

    This class runs LeRobot policy inference in a separate process and communicates with the
    environment to generate actions based on observations. It is optimized for edge devices
    like Raspberry Pi with potentially limited resources.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict | None = None,
    ):
        """Initialize the Pi05 policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
        """
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.overrides = overrides if overrides is not None else {}

        # Extract serializable env data for spawn multiprocessing
        env_metadata = env.get_metadata() if hasattr(env, 'get_metadata') else {}

        context = mp.get_context('spawn')
        self.inf_proc = context.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env_metadata": env_metadata,
                "observation_space": env.observation_space,
                "overrides": self.overrides,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self, task_description: str) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""

        logger.info(f"Syncing task to worker: {task_description}")
        self.parent_conn.send(task_description)

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """
            logger.debug("Requesting action from Pi05 policy...")
            obs_raw: Observation = self.env.get_obs()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            LOG_PATH_OBS = os.path.expanduser("~/crisp_gym_debug/crisp_gym_ITR/obs_log.txt")
            # Inside your _fn():
            with open(LOG_PATH_OBS, "a") as f:
                obs_str = ",".join(map(str, obs_raw["observation.state"].flatten()))
                f.write(f"{obs_str}\n")
                f.flush()  # Force the OS to write to disk immediately

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()

            LOG_PATH = os.path.expanduser("~/crisp_gym_debug/crisp_gym_ITR/actions_log.txt")
            # Inside your _fn():
            with open(LOG_PATH, "a") as f:
                action_str = ",".join(map(str, action.flatten()))
                f.write(f"{action_str}\n")
                f.flush()  # Force the OS to write to disk immediately
            #logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env_metadata: dict,
    observation_space: Any,
    overrides: dict | None = None,
):  # noqa: ANN001
    """Policy inference process optimized for Pi05: loads policy, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        env_metadata (dict): Environment metadata for validation.
        observation_space (Any): Observation space from the environment.
        overrides (dict | None): Optional overrides for the policy configuration.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Pi05 Inference] Could not import third-party plugins for LeRobot. Continuing without them."
        )
    logger.info("[Pi05 Inference] Starting Pi05 inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Pi05 Inference] Using device: {device}")

        logger.info(f"[Pi05 Inference] Loading training config from {pretrained_path}...")

        # train_config = TrainPipelineConfig.from_pretrained(pretrained_path)

        # _check_dataset_metadata(train_config, env_metadata, logger)

        # logger.info("[Pi05 Inference] Loaded training config.")

        # logger.debug(f"[Pi05 Inference] Train config: {train_config}")

        # if train_config.policy is None:
        #     raise ValueError(
        #         f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
        #         "Please ensure the policy is correctly configured."
        #     )
    
        logger.info("[Pi05 Inference] Loading policy...")
        policy_cls = get_policy_class("pi05")
        logger.info("Step 1: Loading Base Model to RAM...")

        policy = policy_cls.from_pretrained(
            pretrained_path,  # We load the base model from the local path where we downloaded it, not from the pretrained_path which is where the adapter is
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
        )

        # policy.config.use_cuda_graphs = False  # Disable CUDA graphs for Pi05 inference
        # if hasattr(policy.config, "compile_model"):
        #     policy.config.compile_model = False

        for override_key, override_value in (overrides or {}).items():
            logger.warning(
                f"[Pi05 Inference] Overriding policy config: {override_key} = {getattr(policy.config, override_key)} -> {override_value}"
            )
            setattr(policy.config, override_key, override_value)

        policy.reset()
        policy.to(device).eval()

        warmup_obs_raw = observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)

        # 2. MANUALLY LOAD THE TOKENIZER
        # Since policy.tokenizer is missing, we load it from the same path
        logger.info("[Pi05 Inference] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224", 
            trust_remote_code=True
        )
        default_text = "complete the task"
    # 3. Define a helper to tokenize text
        def add_language_instruction(obs_dict, text):
            # Pi0 policies have a built-in tokenizer
            tokens = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=200
            ).to(device)
            obs_dict["observation.language.tokens"] = tokens["input_ids"]
            obs_dict["observation.language.attention_mask"] = tokens["attention_mask"].bool()
            return obs_dict

        warmup_obs = add_language_instruction(warmup_obs, default_text)

        logger.info("[Pi05 Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            import time

            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed = end - start
                elapsed_list.append(elapsed)

            torch.cuda.synchronize() if torch.cuda.is_available() else None if torch.cuda.is_available() else None

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Pi05 Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Pi05 Inference] Warm-up complete")
        current_task = default_text
        while True:
            msg = conn.recv()
            if msg is None:
                break
            if msg == "reset":
                logger.info("[Pi05 Inference] Resetting policy")
                policy.reset()
                continue

            if isinstance(msg, str):
                current_task = msg
                logger.info(f"[Pi05 Inference] Task updated to: '{current_task}'")
                # We don't run inference yet, we just wait for the first observation
                continue

            with torch.inference_mode():
                obs = numpy_obs_to_torch(msg)
                
                # ADD THIS LINE HERE:
                obs = add_language_instruction(obs, current_task)
                
                action = policy.select_action(obs)

            # log computed action for visibility
            logger.info(f"[Pi05 Inference] Computed action: {action}")
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Pi05 Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Pi05 Inference] Worker shutting down")


def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env_metadata: dict,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env_metadata (dict): The environment metadata dictionary to compare against.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Pi05 Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Pi05 Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Pi05 Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Pi05 Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Pi05 Inference] Could not load dataset metadata: {e}")
        logger.info("[Pi05 Inference] Skipping metadata comparison.")
