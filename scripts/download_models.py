from huggingface_hub import snapshot_download

# This downloads the entire repo to a specific folder
snapshot_download(
    repo_id="lerobot/pi05_base", 
    local_dir="./models/pi05_base",
)