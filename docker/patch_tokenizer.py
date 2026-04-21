import json
import os
import time
from pathlib import Path

# We define the specific path you identified, but also the base cache dir
SPECIFIC_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/tokenizer_config.json")
BASE_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

def patch_file(file_path):
    """Safely applies the fix_mistral_regex patch to a specific file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Check if it needs patching
        if data.get("fix_mistral_regex") is not True:
            data["fix_mistral_regex"] = True
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✨ Successfully patched: {file_path}")
            return True
    except Exception as e:
        print(f"⚠️ Error patching {file_path}: {e}")
    return False

def run_patch():
    print(f"🔍 Groot Patcher Active...")
    print(f"🎯 Targeting: {SPECIFIC_PATH}")
    
    while True:
        # 1. Check the specific path first
        if os.path.exists(SPECIFIC_PATH):
            patch_file(SPECIFIC_PATH)
        
        # 2. Also scan the base directory in case the structure is different
        # This handles nested 'snapshots' or different 'hub' vs 'lerobot' folders
        for root, _, files in os.walk(BASE_CACHE_DIR):
            if "tokenizer_config.json" in files:
                full_path = os.path.join(root, "tokenizer_config.json")
                # Only patch if it's actually the Groot/Mistral class
                try:
                    with open(full_path, "r") as f:
                        content = f.read()
                        if "Mistral" in content and '"fix_mistral_regex": true' not in content:
                            patch_file(full_path)
                except:
                    pass

        time.sleep(5) # Check every 5 seconds

if __name__ == "__main__":
    run_patch()