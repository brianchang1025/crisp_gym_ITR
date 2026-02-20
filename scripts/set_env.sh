export GIT_LFS_SKIP_SMUDGE=1

# Enable SVT logging (set to 1 to turn on; used by some tools/libraries)
export SVT_LOG=1

# ROS2 DDS domain ID — isolates ROS2 traffic between networks/processes
export ROS_DOMAIN_ID=100
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# CRISP config search path (colon-separated list of directories)
# Replace the example paths with the actual config folders you want to use.
#export CRISP_CONFIG_PATH="/path/to/config1:/path/to/config2"

# Optional: make local package importable (adjust path to your repo root if needed)
#export PYTHONPATH="${PYTHONPATH}:/home/cbrian/crisp_gym"
PATH_CAMERA="/home/cbrian/crisp_gym_ITR/crisp_gym_ITR/.pixi/envs/jazzy-lerobot/lib/python3.12/site-packages/crisp_py/config/cameras"  # optional
PATH_GRIPPER="/home/cbrian/crisp_gym_ITR/crisp_gym_ITR/.pixi/envs/jazzy-lerobot/lib/python3.12/site-packages/crisp_py/config/grippers"  # optional
PATH_CONTROL="/home/cbrian/crisp_gym_ITR/crisp_gym_ITR/.pixi/envs/jazzy-lerobot/lib/python3.12/site-packages/crisp_py/config/control"  # optional
PATH_ROBOT="/home/cbrian/crisp_gym_ITR/crisp_gym_ITR/.pixi/envs/jazzy-lerobot/lib/python3.12/site-packages/crisp_py/config/robots"  # optional
PATH_SENSOR="/home/cbrian/crisp_gym_ITR/crisp_gym_ITR/.pixi/envs/jazzy-lerobot/lib/python3.12/site-packages/crisp_py/config/sensors"  # optional
CRISP_CONFIG_PATH="${PATH_CAMERA}:${PATH_GRIPPER}:${PATH_CONTROL}:${PATH_ROBOT}:${PATH_SENSOR}"
export CRISP_CONFIG_PATH