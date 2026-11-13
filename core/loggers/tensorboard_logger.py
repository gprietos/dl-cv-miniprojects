from typing import Any
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loggers.base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
            
    def log(self, metrics_dict, step):    
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(key, value.item(), step)
            elif isinstance(value, float):   
                self.writer.add_scalar(key, value, step)
                
    def log_params(self, params: dict[str, Any]) -> None:
        # Tensorboard converts dict items to text summaries for display
        self.writer.add_text("hyperparameters", str(params))

    def log_image(self, image: np.ndarray, image_name: str) -> None:
        self.writer.add_image(image_name, image, dataformats="HWC")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()