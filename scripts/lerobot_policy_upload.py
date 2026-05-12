import os
from huggingface_hub import HfApi

REPO_ID = "cbrian/policy_pi05_libero_env_SST_SP_WC1_TC1_dataset_MD1_ctrl_cartesian" 
LOCAL_FOLDER_PATH = "/workspace/outputs/policy_pi05_libero_env_SST_SP_WC1_TC1_dataset_MD1_ctrl_cartesian/checkpoints/last/pretrained_model"  # Path to your trained policy folder

def upload_policy():
    # --- Configuration ---
    # Replace with your HF username and desired repo name
    
    api = HfApi()

    # 1. Ensure the repo exists (uses cached token automatically)
    print(f"Verifying repository: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # 2. Upload the entire folder
    print(f"Uploading files from {LOCAL_FOLDER_PATH}...")
    api.upload_folder(
        folder_path=LOCAL_FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update trained policy from ROS 2 node"
    )

    print(f"Done! Check it out at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    if os.path.exists(LOCAL_FOLDER_PATH):
        upload_policy()
    else:
        print("Error: Local folder not found. Check your path!")