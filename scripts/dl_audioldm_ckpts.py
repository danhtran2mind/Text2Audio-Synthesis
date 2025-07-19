from huggingface_hub import snapshot_download
import os
import shutil

# Download checkpoints folder
snapshot_download(
    repo_id="ayousanz/AudioLDM-training-finetuning",
    allow_patterns="checkpoints/*",
    local_dir="./ckpts",
    repo_type="model"
)

# Move files from checkpoints to ckpts and remove empty folder
source_dir = "./ckpts/checkpoints"
dest_dir = "./ckpts"
for file in os.listdir(source_dir):
    shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))
os.rmdir(source_dir)
