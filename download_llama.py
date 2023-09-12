from huggingface_hub import snapshot_download
import os
import sys

def download_repository(repo_id, target_folder, revision="main", token=None):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the repository snapshot
    snapshot_path = snapshot_download(repo_id=repo_id, local_dir=target_folder, local_dir_use_symlinks=False, revision=revision, token=token)

if __name__ == "__main__":
    # Take repository ID as an argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py repository_id")
        sys.exit(1)
    
    repository_id = sys.argv[1]

    # Extract the second part of the repository ID
    repo_id_part = repository_id.split('/')[1]

    # Specify the target folder and token
    target_folder = f"/scratch/yerong/.cache/pyllama/{repo_id_part}/"
    huggingface_token = "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"

    # Download the repository snapshot
    download_repository(repository_id, target_folder, token=huggingface_token)
