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


def download_and_extract_dataset():
    """Download and extract the MusicBench dataset."""
    os.makedirs("data/audioset/music_bench", exist_ok=True)
    snapshot_download(repo_id="amaai-lab/MusicBench", cache_dir="data",
                      repo_type="dataset")
    raw_data_dir = "data/datasets--amaai-lab--MusicBench"
    snapshot_dir = os.path.join(raw_data_dir, "snapshots",
                                os.listdir(os.path.join(raw_data_dir, "snapshots"))[0])
    music_bench_dir = "data/audioset/music_bench"

    with tarfile.open(f"{snapshot_dir}/MusicBench.tar.gz", "r:gz") as tar:
        tar.extractall(path=music_bench_dir)
    return raw_data_dir, music_bench_dir


def move_and_cleanup_files(raw_data_dir, music_bench_dir):
    """Move files from datashare to music_bench and clean up unnecessary directories."""
    datashare_dir = os.path.join(music_bench_dir, "datashare")
    for item in os.listdir(datashare_dir):
        shutil.move(os.path.join(datashare_dir, item), music_bench_dir)
    shutil.rmtree(datashare_dir)
    shutil.rmtree(raw_data_dir)


def prepare_json_data(train_df, val_df, music_bench_dir):
    """Prepare JSON data for train and validation sets."""
    train_data = [
        {"wav": os.path.join(music_bench_dir, row["location"]),
         "caption": row["main_caption"]}
        for _, row in train_df.iterrows()
    ]
    val_data = [
        {"wav": os.path.join(music_bench_dir, row["location"]),
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
                    "val": "./data/audioset/val.json",
                    "class_label_indices": "../metadata/audiocaps/class_labels_indices.csv"
                }
            }
        }
    }
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/dataset_root.json", "w") as f:
        json.dump(dataset_root, f, indent=4)


def main():
    """Main function to execute the dataset processing pipeline."""
    train_df, val_df = load_and_clean_dataset()
    raw_data_dir, music_bench_dir = download_and_extract_dataset()
    move_and_cleanup_files(raw_data_dir, music_bench_dir)
    train_data, val_data = prepare_json_data(train_df, val_df, music_bench_dir)
    write_json_files(train_data, val_data)
    create_dataset_root_json()


if __name__ == "__main__":
    main()
