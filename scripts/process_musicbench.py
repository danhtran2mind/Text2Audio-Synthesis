from datasets import load_dataset
import pandas as pd
from huggingface_hub import snapshot_download
import os
import shutil
import tarfile
import json

# Load and clean dataset
dataset = load_dataset("amaai-lab/MusicBench")
train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['test'])

# Remove duplicates by keeping row with longest 'main_caption' per 'location'
train_cleaned_df = train_df.loc[train_df.groupby('location')['main_caption'].apply(lambda x: x.str.len().idxmax())]
val_cleaned_df = val_df.loc[val_df.groupby('location')['main_caption'].apply(lambda x: x.str.len().idxmax())]

print(f"Train set size: {len(train_cleaned_df)}, Validation set size: {len(val_cleaned_df)}")

# Download dataset
os.makedirs("data", exist_ok=True)
snapshot_download(repo_id="amaai-lab/MusicBench", cache_dir="data", repo_type="dataset")

# Set up directories
os.chdir("data")
raw_data_dir = "datasets--amaai-lab--MusicBench"
music_bench_dir = "./audioset/music_bench"
snapshots_dir = os.path.join(raw_data_dir, "snapshots")
snapshot_folder = os.listdir(snapshots_dir)[0]

os.makedirs(os.path.join("audioset", "music_bench"), exist_ok=True)
os.makedirs(os.path.join("audioset", "music_bench_test"), exist_ok=True)

# Extract MusicBench.tar.gz
with tarfile.open(f"{snapshots_dir}/{snapshot_folder}/MusicBench.tar.gz", "r:gz") as tar:
    tar.extractall(path=music_bench_dir)

# Move contents from datashare to music_bench
datashare_dir = os.path.join(music_bench_dir, "datashare")
for item in os.listdir(datashare_dir):
    shutil.move(os.path.join(datashare_dir, item), os.path.join(music_bench_dir, item))
os.rmdir(datashare_dir)

# Clean up
shutil.rmtree(raw_data_dir)

# Create dataset_root.json
data = {
    "audiocaps": "./data/audioset",
    "comments": {},
    "metadata": {
        "path": {
            "audiocaps": {
                "train": "./data/audioset/train.json",
                "test": "./data/audioset/test.json",
                "val": "./data/audioset/val.json",
                "class_label_indices": "../metadata/audiocaps/class_labels_indices.csv"
            }
        }
    }
}

os.makedirs("metadata", exist_ok=True)
with open("metadata/dataset_root.json", "w") as f:
    json.dump(data, f, indent=4)
