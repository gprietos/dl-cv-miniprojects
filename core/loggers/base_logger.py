from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch

class BaseLogger(ABC):
    
    @abstractmethod
    def log(self, metrics_dict: dict[str, Any], step: int) -> None:
        """Log scalar metrics at a given training epoch or step."""
        pass
        
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log config hyperparameters and reproducibility metadata."""
        pass

    @abstractmethod
    def log_image(self, image: np.ndarray, image_name: str) -> None:        
        """Log visual metrics like confusion matrices or visualizations."""
        pass

    @abstractmethod
    def save_model_state(self, state_dict: dict[str, Any], filename: str) -> None:
        """
        Save full training state (weights, optimizers, schedulers) to resume training.
        Default is a pass-through for loggers that don't support file storage.
        """
        pass

    @abstractmethod
    def save_model(self, model: torch.nn.Module, filename: str) -> None:
        """
        Save only the clean model architecture and weights for production inference.
        Default is a pass-through for loggers that don't support file storage.
        """
        pass

    @abstractmethod
    def close(self) -> None: 
        """Flushes buffers and safely wraps up active connections/runs."""
        pass