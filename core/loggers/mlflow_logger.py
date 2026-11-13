import mlflow
import mlflow.artifacts
import mlflow.pytorch
from typing import Any

import torch
from core.loggers.base_logger import BaseLogger


class MLFlowLogger(BaseLogger):
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str = None,
        run_id: str = None,
        tags: dict[str, Any] = None,
        description: str = None,
        log_system_metrics: bool = True,
    ):
        """
        Initialize the MLFlow logger.

        Args:
            tracking_uri:       URI of the MLFlow tracking server.
            experiment_name:    Name of the MLFlow experiment.
            run_name:           Optional display name for a new run.
            run_id:             Optional existing run ID to resume. Takes precedence over run_name.
            tags:               Optional key-value metadata attached to the run at creation time
                                (e.g. git commit hash, dataset version, hostname).
            description:        Optional human-readable description of what the run is testing.
            log_system_metrics: If True, automatically logs CPU/GPU/memory usage. Defaults to True.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = run_id
        self.tags = tags
        self.description = description
        self.log_system_metrics = log_system_metrics
        self.artifact_uri = None
        self._active_run = None
        self._artifact_cache: set[str] = set()

        self._configure_mlflow()
        self._start_run(run_name=run_name, run_id=run_id, tags=tags, description=description)


    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _configure_mlflow(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.environment_variables.MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR = False

    def _start_run(
        self,
        run_name: str = None,
        run_id: str = None,
        tags: dict[str, Any] = None,
        description: str = None,
    ) -> None:
        """Open and hold an MLFlow run for the lifetime of this logger."""
        self._active_run = mlflow.start_run(
            run_id=run_id,
            run_name=run_name,
            tags=tags,
            description=description,
            log_system_metrics=self.log_system_metrics,
        )
        self.run_id = self._active_run.info.run_id
        self.artifact_uri = self._active_run.info.artifact_uri

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, metrics_dict: dict[str, Any], step: int) -> None:
        """
        Log a dictionary of metrics at a given step.

        Args:
            metrics_dict: Mapping of metric name to value.
            step:         Training step or epoch number.
        """

        scalar_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (float, int))}
        if scalar_metrics:
            mlflow.log_metrics(scalar_metrics, step=step, run_id=self.run_id)


    def log_params(self, params: dict) -> None:
        """
        Log hyperparameters and config values for the run.
        Should be called once at the start of training.

        Args:
            params: Flat mapping of parameter name to value.
        """

        mlflow.log_params(params, run_id=self.run_id)


    def log_summary(self, summary_dict: dict[str, Any], filename: str = "best_metrics") -> None:
        try:
            mlflow.log_dict(summary_dict, f"{filename}.json", run_id=self.run_id)
            print(f"[MLFlowLogger] Successfully logged summary dictionary to '{filename}.json'")
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log summary dict: {e}")


    def log_image(self, image, image_name: str) -> None:
        """
        Log an image artifact under the 'images/' folder.

        Args:
            image_name: File name (e.g. 'confusion_matrix.png').
            image:      Image object accepted by mlflow.log_image.
        """
        try:
            mlflow.log_image(image, artifact_file=f"images/{image_name}")
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log image '{image_name}': {e}")


    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_model_state(self, state_dict: dict[str, Any], filename: str) -> None:
        try:
            mlflow.pytorch.log_state_dict(
                state_dict=state_dict, 
                artifact_path=f"checkpoints/{filename}", 
            )
            print(f"[MLFlowLogger] Logged training recovery state to 'checkpoints/{filename}'")
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log training state '{filename}': {e}")


    def save_model(self, model: torch.nn.Module, filename: str) -> None:
        try:
            # Strip compiled model wrapper if it exists so production weights are clean
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            
            mlflow.pytorch.log_model(
                pytorch_model=raw_model, 
                name=filename,
                registered_model_name=None,  # Avoid automatic registration to MLFlow Model Registry
            )
            print(f"[MLFlowLogger] Registered inference model to production artifact: 'models/{filename}'")
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log production model '{filename}': {e}")



    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """End the active MLFlow run. Should be called after training completes."""
        if self._active_run is not None:
            mlflow.end_run()
            self._active_run = None


    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "active" if self._active_run is not None else "closed"
        return (
            f"MLFlowLogger("
            f"experiment='{self.experiment_name}', "
            f"run_id='{self.run_id}', "
            f"status={status}, "
            f"tracking_uri='{self.tracking_uri}')"
        )