from huggingface_hub import snapshot_download
import os
import shutil
import argparse

def download_checkpoints(local_dir):
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download checkpoints folder from Hugging Face Hub
    try:
        snapshot_download(
            repo_id="ayousanz/AudioLDM-training-finetuning",
            allow_patterns="checkpoints/*",
            local_dir=local_dir,
            repo_type="model"
        )
        print(f"Successfully downloaded checkpoints to {local_dir}")
    except Exception as e:
        print(f"Error downloading checkpoints: {e}")
        raise

def move_and_clean_checkpoints(local_dir):
    # Define source and destination directories
    source_dir = os.path.join(local_dir, "checkpoints")
    dest_dir = local_dir

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist. Check if checkpoints were downloaded.")
        return

    # Move files from checkpoints to local_dir
    try:
        for file in os.listdir(source_dir):
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(dest_dir, file)
            # Ensure no overwrite conflicts
            if os.path.exists(dest_path):
                print(f"Skipping {file}: already exists in {dest_dir}")
                continue
            shutil.move(source_path, dest_path)
            print(f"Moved {file} to {dest_dir}")
        
        # Remove the empty checkpoints directory
        os.rmdir(source_dir)
        print(f"Removed empty directory: {source_dir}")
    except Exception as e:
        print(f"Error moving files or removing directory: {e}")
        raise

def model_checkpoint_process(local_dir):
    download_checkpoints(local_dir)
    move_and_clean_checkpoints(local_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and move checkpoints from Hugging Face Hub.")
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./ckpts/AudioLDM",
        help="Local directory to download and store checkpoints"
    )
    args = parser.parse_args()
    
    model_checkpoint_process(args.local_dir)
