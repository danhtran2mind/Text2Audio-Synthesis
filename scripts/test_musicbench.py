from datasets import load_dataset
import pandas as pd
from huggingface_hub import snapshot_download
import os
import shutil
import tarfile
import json
import multiprocessing as mp
import argparse


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


def move_file(args):
    """Helper function to move a single file and handle errors."""
    src_path, dst_path, index = args
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        return index, True
    else:
        print(f"Warning: File not found: {src_path}")
        return index, False
        

def move_and_cleanup_files(raw_data_dir, music_bench_dir, train_df, val_df, num_processes):
    """Move files from datashare to music_bench, organize into train/test folders, and clean up."""
    # Create train and test directories
    os.makedirs("data/audioset/train", exist_ok=True)
    os.makedirs("data/audioset/test", exist_ok=True)
    
    datashare_dir = os.path.join(music_bench_dir, "datashare")

    # Prepare arguments for parallel file moving
    train_tasks = [
        (os.path.join(datashare_dir, row["location"]),
         os.path.join("data/audioset/train", os.path.basename(row["location"])),
         index)
        for index, row in train_df.iterrows()
    ]
    val_tasks = [
        (os.path.join(datashare_dir, row["location"]),
         os.path.join("data/audioset/test", os.path.basename(row["location"])),
         index)
        for index, row in val_df.iterrows()
    ]

    # Use multiprocessing pool for file moving
    with mp.Pool(processes=num_processes) as pool:
        # Process train files
        train_results = pool.map(move_file, train_tasks)
        # Drop rows where files were not found
        for index, success in train_results:
            if not success:
                train_df = train_df.drop(index)
        
        # Process validation files
        val_results = pool.map(move_file, val_tasks)
        # Drop rows where files were not found
        for index, success in val_results:
            if not success:
                val_df = val_df.drop(index)

    # Remove files in train folder not in train_df
    train_filenames = set(os.path.basename(row["location"]) for _, row in train_df.iterrows())
    train_dir = "data/audioset/train"
    for filename in os.listdir(train_dir):
        if filename not in train_filenames:
            file_path = os.path.join(train_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed extra file from train folder: {file_path}")

    # Remove files in test folder not in val_df
    test_filenames = set(os.path.basename(row["location"]) for _, row in val_df.iterrows())
    test_dir = "data/audioset/test"
    for filename in os.listdir(test_dir):
        if filename not in test_filenames:
            file_path = os.path.join(test_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed extra file from test folder: {file_path}")

    # Clean up datashare and raw data directories
    if os.path.exists(music_bench_dir):
        shutil.rmtree(music_bench_dir)
    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    
    return train_df, val_df


def prepare_json_data(train_df, val_df):
    """Prepare JSON data for train and validation sets with updated paths."""
    train_data = [
        {"wav": os.path.join("data/audioset/train", os.path.basename(row["location"])),
         "caption": row["main_caption"]}
        for _, row in train_df.iterrows()
    ]
    val_data = [
        {"wav": os.path.join("data/audioset/test", os.path.basename(row["location"])),
         "caption": row["main_caption"]}
        for _, row in val_df.iterrows()
    ]
    return train_data, val_data


def write_json_files(train_data, val_data):
    """Write train and validation data to JSON files."""
    os.makedirs("data/audioset", exist_ok=True)
    with open("data/audioset/train.json", "w") as f:
        json.dump({"data": train_data}, f, indent=4)
    with open("data/audioset/test.json", "w") as f:
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
                    "test": "./data/audioset/test.json",
                    "class_label_indices": "../metadata/audiocaps/class_labels_indices.csv"
                }
            }
        }
    }
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/dataset_root.json", "w") as f:
        json.dump(dataset_root, f, indent=4)


def main(arg_process):
    """Main function to execute the dataset processing pipeline."""
    # Limit the number of processes to the minimum of arg_process and CPU count
    num_processes = min(arg_process, os.cpu_count())
    print(f"Using {num_processes} processes for parallel file operations.")

    raw_data_dir = "data/datasets--amaai-lab--MusicBench"
    music_bench_dir = "./data/audioset/music_bench"

    train_df, val_df = load_and_clean_dataset()
    download_and_extract_dataset(raw_data_dir, music_bench_dir)
    train_df, val_df = move_and_cleanup_files(raw_data_dir, music_bench_dir, train_df, val_df, num_processes)
    train_data, val_data = prepare_json_data(train_df, val_df)
    write_json_files(train_data, val_data)
    create_dataset_root_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MusicBench dataset with parallel file operations.")
    parser.add_argument("--arg_process", type=int, default=os.cpu_count(),
                        help="Number of processes to use (capped at CPU count).")
    args = parser.parse_args()
    main(args.arg_process)
