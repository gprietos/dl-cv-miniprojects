import os
import yaml
import torch
import numpy as np
import random
import subprocess


def set_seed(seed=42):
    """ Set the seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


def load_config_dict(configuration_path):
    """
    Creates a configuration dictionary from a yaml configuration file
    """
    with open(configuration_path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_dvc_dataset_hash(dvc_file_path: str) -> str:
    """Reads the MD5 hash from a .dvc file to track the exact data version."""
    if not os.path.exists(dvc_file_path):
        return "no_dvc_file_found"
    try:
        with open(dvc_file_path, "r") as f:
            dvc_meta = yaml.safe_load(f)
        # DVC structures files by mapping targets out under the 'outs' key
        return dvc_meta["outs"][0]["md5"]
    except Exception as e:
        return f"error_reading_dvc: {str(e)}"