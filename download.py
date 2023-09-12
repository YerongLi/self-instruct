from huggingface_hub import snapshot_download
import os

def download_repository(repo_id, target_folder, revision="main", token=None):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the repository snapshot
    snapshot_path = snapshot_download(repo_id=repo_id, local_dir=target_folder,local_dir_use_symlinks=False, revision=revision, token=token)



if __name__ == "__main__":
    # Specify the repository ID, target folder, and token
    repository_id = "TheBloke/Llama-2-7B-GPTQ"
    target_folder = "/scratch/yerong/.cache/pyllama/Llama-2-7B-GPTQ/"
    huggingface_token = "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"

    # Download the repository snapshot
    download_repository(repository_id, target_folder, token=huggingface_token)

