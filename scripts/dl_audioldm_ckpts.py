from huggingface_hub import snapshot_download
import os
import shutil
import argparse

def download_and_move_checkpoints(local_dir):
    # Download checkpoints folder
    snapshot_download(
        repo_id="ayousanz/AudioLDM-training-finetuning",
        allow_patterns="checkpoints/*",
        local_dir=local_dir,
        repo_type="model"
    )

    # Move files from checkpoints to local_dir and remove empty folder
    source_dir = os.path.join(local_dir, "checkpoints")
    dest_dir = local_dir
    for file in os.listdir(source_dir):
        shutil.move(
            os.path.join(source_dir, file),
            os.path.join(dest_dir, file)
        )
    os.rmdir(source_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and move checkpoints from Hugging Face Hub.")
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./ckpts",
        help="Local directory to download and store checkpoints"
    )
    args = parser.parse_args()
    
    download_and_move_checkpoints(args.local_dir)
