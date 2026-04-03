from huggingface_hub import snapshot_download

# This downloads the entire repo to a specific folder
snapshot_download(
    repo_id="cbrian/pi05_pretrain_libero_dataset_E3_cartesian", 
    local_dir="./models/pi05_pretrain_libero_dataset_E3_cartesian",
)
# snapshot_download(
#     repo_id="cbrian/pi05joint", 
#     local_dir="./models/pi05_joint",
# )