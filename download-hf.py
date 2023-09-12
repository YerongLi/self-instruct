from huggingface_hub import snapshot_download
import os

def download_repository(repo_id, target_folder, revision="main", token=None):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the repository snapshot
    snapshot_path = snapshot_download(repo_id=repo_id, revision=revision, token=token)

    # Move the downloaded snapshot to the target folder
    target_path = os.path.join(target_folder, os.path.basename(snapshot_path))
    os.rename(snapshot_path, target_path)

    print(f"Repository snapshot downloaded to: {target_path}")

if __name__ == "__main__":
    # Specify the repository ID, target folder, and token
    repository_id = "meta-llama/Llama-2-70b-hf"
    target_folder = "/scratch/yerong/.cache/pyllama/Llama-2-70b-hf/"
    huggingface_token = "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"

    # Download the repository snapshot
    download_repository(repository_id, target_folder, token=huggingface_token)

