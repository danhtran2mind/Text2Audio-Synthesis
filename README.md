# Text to Audio AudioLDM (Text2Audio AudioLDM)

## Download Sample AudioLDM checkpoints
The code for download and process AudioLDM checkpoints available at: `scripts/dl_audioldm_ckpts.py`. Just run:
```bash
python scripts/dl_audioldm_ckpts.py
```
The checkpoints will stay at `ckpts` folder.

## Dataset
The processing dataset at `scripts/process_musicbench.py`
```bash
python scripts/process_musicbench.py
```
The Processed data will stay at `data` folder.
## Training

```bash
git clone -b dev https://github.com/danhtran2mind/AudioLDM-training-finetuning.git
cd AudioLDM-training-finetuning
# Install running environment

pip install -q torchlibrosa ftfy braceexpand webdataset 
pip install -q wget taming-transformers
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
cd data/dataset/audioset
unzip --quiet train.zip
!unzip --quiet val.zip
unzip --quiet test.zip
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
python src/audioldm/train.py \
    --config_yaml configs/AudioLDM_training_configs/audioldm_original.yaml \
    --reload_from_ckpt ckpts/audioldm-s-full.ckpt \
    --accelerator gpu \
    --wandb_off
```

## Inference

### Generation mode
```bash
python src/audioldm/infer.py \
    --mode generation \
    --text "<your_text_prompt>" \
    --save_dir "<your_output_dir>" \
    --prompt_as_filename \
    # --model_name "$MODEL_NAME" \
    --batchsize <batch_size> \
    --ddim_steps <ddim_steps> \ # Should be scaled between 50 and 500.
    --guidance_scale <guidance_scale> \
    --duration <duration_of_output_audio_in_second>> \
    --n_candidate_gen_per_text <num_candidates> \
    --seed <random_seed>
```
### Transfer mode

```bash
python src/audioldm/infer.py \
    --mode transfer \
    --text ""<your_text_prompt>"" \
    --file_path "$INPUT_AUDIO" \
    --transfer_strength <transfer_strength> \ # Should be be a float in the range [0.0, 1.0] 
    --save_dir "<your_output_dir>" \
    --prompt_as_filename \
    # --model_name "$MODEL_NAME" \
    --batchsize <batch_size> \
    --ddim_steps <ddim_steps> \ # Should be scaled between 50 and 500.
    --guidance_scale <guidance_scale> \
    --duration <duration_of_output_audio_in_second>> \
    --n_candidate_gen_per_text <num_candidates> \
    --seed <random_seed>
```


## Reference
This project uses code from the following sources:
- `haoheliu/AudioLDM` (https://github.com/haoheliu/AudioLDM)
- `haoheliu/AudioLDM-training-finetuning` (https://github.com/haoheliu/AudioLDM-training-finetuning)
