import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(__file__))
from audioldm_train.train.latent_diffusion import train  # Relative import

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=True,
        help="path to config .yaml file",
    )
    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="perform validation",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="accelerator type: gpu or cpu",
    )
    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="disable Wandb logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility, default is 42",
    )

    args = parser.parse_args()

    perform_validation = args.val
    accelerator = args.accelerator
    wandb_off = args.wandb_off
    seed = args.seed

    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, use --accelerator cpu instead")

    config_yaml = args.config_yaml
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"]["crossattn_audiomae_generated"]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    train(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation, accelerator, wandb_off, seed)
