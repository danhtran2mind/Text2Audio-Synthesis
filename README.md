# Text to Audio Synthesis (Text2Audio Synthesis)

## Dataset
The processing dataset at `scripts/process_musicbench.py`
```bash
python scripts/process_musicbench.py
```
The Processed data will stay at `data` folder
## Training

```bash
%cd /content
!git clone -b dev https://github.com/danhtran2mind/AudioLDM-training-finetuning.git
%cd AudioLDM-training-finetuning
# Install running environment

!pip install -q torchlibrosa ftfy braceexpand webdataset 
!pip install -q wget taming-transformers
```

```python
from huggingface_hub import snapshot_download
import os

# Define the repository ID and destination folder
repo_id = "heboya8/my-AudioLDM"
local_dir = "data/dataset"

# Create the data folder if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download the entire repository
print(f"Downloading repository {repo_id}...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    # local_dir_use_symlinks=False,
    # force_download=True,
    repo_type="dataset"
)
print(f"Repository downloaded to {local_dir}")
```

```bash
%cd data/dataset/audioset
!unzip --quiet train.zip
!unzip --quiet val.zip
!unzip --quiet test.zip
```
```python
from huggingface_hub import snapshot_download

# Download only the checkpoints folder
snapshot_download(
    repo_id="ayousanz/AudioLDM-training-finetuning",
    allow_patterns="checkpoints/*",  # Only download files in the checkpoints folder
    local_dir="./data",  # Local directory to save the files
    repo_type="model"  # Specify repo type as dataset since it's not a model
)
```
```bash
!python audioldm_train/train/latent_diffusion.py \
    --config_yaml audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
    --reload_from_ckpt data/checkpoints/audioldm-s-full.ckpt \
    --wandb_off
```

## TO DO
- [ ] Fix path at:
    - [ ] `src/audioldm/audioldm_train/modules/latent_diffusion/ddpm.py` line 113
    - [ ] `src/audioldm/audioldm_train/utilities/model_util.py` line 267
- [ ] Create sperated `src/audioldm/train.py`
- [ ] add num_gradient_accum

## Reference
This project use code from:
- `haoheliu/AudioLDM` (https://github.com/haoheliu/AudioLDM)
- `haoheliu/AudioLDM-training-finetuning` (https://github.com/haoheliu/AudioLDM-training-finetuning)
