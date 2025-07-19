from datasets import load_dataset
import pandas as pd
from huggingface_hub import snapshot_download
import os
import shutil
import tarfile
import json


def load_and_clean_dataset():
    """Load and clean the MusicBench dataset, removing duplicates by keeping the row
    with the longest main_caption per location."""
    dataset = load_dataset("amaai-lab/MusicBench")
    train_df = pd.DataFrame(dataset["train"]).groupby("location")["main_caption"].apply(
        lambda x: x.loc[x.str.len().idxmax()]
    ).reset_index()
    val_df = pd.DataFrame(dataset["test"]).groupby("location")["main_caption"].apply(
        lambda x: x.loc[x.str.len().idxmax()]
    ).reset_index()
    print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
    return train_df, val_df


def download_and_extract_dataset(raw_data_dir, music_bench_dir):
    """Download and extract the MusicBench dataset."""
    os.makedirs("data/audioset/music_bench", exist_ok=True)
    snapshot_download(repo_id="amaai-lab/MusicBench", cache_dir="data",
                      repo_type="dataset")

    snapshot_dir = os.path.join(raw_data_dir, "snapshots",
                                os.listdir(os.path.join(raw_data_dir, "snapshots"))[0])
    
    with tarfile.open(f"{snapshot_dir}/MusicBench.tar.gz", "r:gz") as tar:
        tar.extractall(path=music_bench_dir)


def move_and_cleanup_files(raw_data_dir, music_bench_dir, train_df, val_df):
    """Move files from datashare to music_bench, organize into train/val folders, and clean up."""
    # Create train and val directories
    os.makedirs("data/audioset/train", exist_ok=True)
    os.makedirs("data/audioset/val", exist_ok=True)
    os.makedirs("data/audioset/test", exist_ok=True)
    # Move train files
    for _, row in train_df.iterrows():
        src_path = os.path.join(music_bench_dir, row["location"])
        # Extract filename from location
        filename = os.path.basename(row["location"])
        dst_path = os.path.join("data/audioset/train", filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: File not found: {src_path}")

    # Move validation files
    for _, row in val_df.iterrows():
        src_path = os.path.join(music_bench_dir, row["location"])
        # Extract filename from location
        filename = os.path.basename(row["location"])
        dst_path = os.path.join("data/audioset/val", filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: File not found: {src_path}")

    # Clean up datashare and raw data directories
    datashare_dir = os.path.join(music_bench_dir, "datashare")
    if os.path.exists(datashare_dir):
        shutil.rmtree(datashare_dir)
    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)


def prepare_json_data(train_df, val_df):
    """Prepare JSON data for train and validation sets with updated paths."""
    train_data = [
        {"wav": os.path.join("data/audioset/train", os.path.basename(row["location"])),
         "caption": row["main_caption"]}
        for _, row in train_df.iterrows()
    ]
    val_data = [
        {"wav": os.path.join("data/audioset/val", os.path.basename(row["location"])),
         "caption": row["main_caption"]}
        for _, row in val_df.iterrows()
    ]
    return train_data, val_data


def write_json_files(train_data, val_data):
    """Write train and validation data to JSON files."""
    os.makedirs("data/audioset", exist_ok=True)
    with open("data/audioset/train.json", "w") as f:
        json.dump({"data": train_data}, f, indent=4)
    with open("data/audioset/val.json", "w") as f:
        json.dump({"data": val_data}, f, indent=4)


def create_dataset_root_json():
    """Create dataset_root.json with metadata configuration."""
    dataset_root = {
        "audiocaps": "./data/audioset",
        "comments": {},
        "metadata": {
            "path": {
                "audiocaps": {
                    "train": "./data/audioset/train.json",
                    "train": "./data/audioset/test.json",
                    "val": "./data/audioset/val.json",
                    "class_label_indices": "/data/metadata/audiocaps/class_labels_indices.csv"
                }
            }
        }
    }
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/dataset_root.json", "w") as f:
        json.dump(dataset_root, f, indent=4)


def main():
    """Main function to execute the dataset processing pipeline."""

    raw_data_dir = "data/datasets--amaai-lab--MusicBench"
    music_bench_dir = "./data/audioset/music_bench/datashare"

    train_df, val_df = load_and_clean_dataset()
    download_and_extract_dataset(raw_data_dir, music_bench_dir)
    move_and_cleanup_files(raw_data_dir, music_bench_dir, train_df, val_df)
    train_data, val_data = prepare_json_data(train_df, val_df)
    write_json_files(train_data, val_data)
    create_dataset_root_json()


if __name__ == "__main__":
    main()
