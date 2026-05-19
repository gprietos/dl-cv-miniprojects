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