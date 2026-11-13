import os
from typing import Any, Optional, List
import torch
from core.loggers.base_logger import BaseLogger


class CheckpointManager:
    def __init__(
        self,
        loggers: List[BaseLogger],
        monitored_metric: str = "val_loss",
        mode: str = "min",
        every_n_epochs: int = 0,
    ):
        """
        Manages training checkpoints and production model routing across multiple loggers.
        """
        self.loggers = loggers
        self.monitored_metric = monitored_metric
        self.mode = mode
        self.every_n_epochs = every_n_epochs
        
        self._validate_config()
        self.best_metric_value = float("inf") if self.mode == "min" else float("-inf")


    def _validate_config(self) -> None:
        if self.mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")
        if self.every_n_epochs < 0:
            raise ValueError("every_n_epochs must be an integer >= 0.")


    def save_periodic_checkpoint(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scaler: Optional[torch.amp.GradScaler] = None,
        scheduler: Optional[Any] = None,
        extra_metrics: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Checks the epoch interval and saves a full recovery training state across all loggers.
        Call this at the very end of your training loop.
        """
        if self.every_n_epochs == 0 or (epoch + 1) % self.every_n_epochs != 0:
            return

        # Strip compiled wrapper to keep weights file clean
        raw_uncompiled_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        
        state_payload = {
            "epoch": epoch + 1,
            "model_state": raw_uncompiled_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if scaler is not None:
            state_payload["scaler_state"] = scaler.state_dict()
        if scheduler is not None:
            state_payload["scheduler_state"] = scheduler.state_dict()
        if extra_metrics is not None:
            state_payload["metrics"] = extra_metrics

        filename = f"epoch_{epoch+1:03d}"
        for logger in self.loggers:
            logger.save_model_state(state_payload, filename=filename)


    def evaluate_best_model(self, epoch: int, metrics_dict: dict[str, Any], model: torch.nn.Module) -> bool:
        """
        Compares current validation metrics against the historical best and saves production weights.
        Call this right after your validation loop computes metrics.
        """
        if self.monitored_metric not in metrics_dict:
            print(f"[Warning] Monitored metric '{self.monitored_metric}' not found in validation metrics. "
                  f"Skipping performance tracking for epoch {epoch+1}.")
            return False

        current_value = metrics_dict[self.monitored_metric]
        is_best = False

        # Check optimization condition (min vs max)
        if (self.mode == "min" and current_value < self.best_metric_value) or \
           (self.mode == "max" and current_value > self.best_metric_value):
            
            print(f"🔥 New best performance achieved! "
                  f"{self.monitored_metric}: {self.best_metric_value:.4f} -> {current_value:.4f}")
            
            self.best_metric_value = current_value
            is_best = True

            # Save clean inference model & metrics summary across all pluggable loggers
            for logger in self.loggers:
                logger.save_model(model, filename="best")
                
                summary_data = {"epoch": epoch + 1, **metrics_dict}
                logger.log_summary(summary_data, filename="best_metrics")

        return is_best