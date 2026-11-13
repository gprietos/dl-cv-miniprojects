import os
from typing import Any
import torch
import json
import numpy as np
from core.loggers.base_logger import BaseLogger


class LocalLogger(BaseLogger):
    def __init__(self, experiment_dir: str, run_name: str):
        self.run_dir = os.path.join(experiment_dir, run_name)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")


    def log_summary(self, summary_dict: dict[str, Any], filename: str = "best_metrics") -> None:
        summary_file = os.path.join(self.run_dir, f"{filename}.json")
        with open(summary_file, "w") as f:
            json.dump(summary_dict, f, indent=4)
        print(f"[LocalLogger] Saved final summary file to {summary_file}")


    def log(self, metrics_dict: dict[str, Any], step: int) -> None:
        payload = {"step": step, **metrics_dict}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(payload) + "\n")


    def log_params(self, params: dict[str, Any]) -> None:
        params_file = os.path.join(self.run_dir, "params.json")
        with open(params_file, "w") as f:
            json.dump(params, f, indent=4)


    def log_image(self, image: np.ndarray, image_name: str) -> None:
        # Optional: save image to disk with matplotlib/PIL if desired
        pass


    def save_modelstate(self, state_dict: dict[str, Any], filename: str) -> None:
        save_path = os.path.join(self.checkpoint_dir, f"{filename}_state.pt")
        torch.save(state_dict, save_path)
        print(f"[LocalLogger] Saved training recovery state to {save_path}")


    def save_model(self, model: torch.nn.Module, filename: str) -> None:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        save_path = os.path.join(self.checkpoint_dir, f"{filename}_weights.pt")
        
        # Local saving stores just the state_dict for clean deployment weights
        torch.save(raw_model.state_dict(), save_path)
        print(f"[LocalLogger] Saved pure inference weights to {save_path}")


    def close(self) -> None:
        print(f"[LocalLogger] Run saved locally at: {self.run_dir}")