import mlflow
import mlflow.artifacts
import mlflow.pytorch
from typing import Any
from loggers.base_logger import BaseLogger


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
    # Artifact helpers
    # ------------------------------------------------------------------

    def _list_artifacts_recursively(self, parent_path: str = "") -> set[str]:
        """
        Recursively list all artifact paths for the current run.
        Results are cached and invalidated only when new artifacts are logged.
        """
        artifacts = mlflow.artifacts.list_artifacts(run_id=self.run_id, artifact_path=parent_path)
        paths: set[str] = set()
        for artifact in artifacts:
            if artifact.is_dir:
                paths |= self._list_artifacts_recursively(artifact.path)
            else:
                paths.add(artifact.path)
        return paths

    def _invalidate_artifact_cache(self) -> None:
        self._artifact_cache = set()

    def _get_artifact_cache(self) -> set[str]:
        if not self._artifact_cache:
            self._artifact_cache = self._list_artifacts_recursively()
        return self._artifact_cache


    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, metrics_dict: dict, step: int) -> None:
        """
        Log a dictionary of metrics at a given step.

        Scalar values (int/float) are batched into a single log_metrics call.
        Non-scalar values are appended to a per-key JSON artifact list.

        Args:
            metrics_dict: Mapping of metric name to value.
            step:         Training step or epoch number.
        """
        self._assert_run_active()
        scalar_metrics: dict[str, float] = {}

        for key, value in metrics_dict.items():
            if isinstance(value, (float, int)):
                scalar_metrics[key] = value
            else:
                try:
                    artifact_key = f"{key}.json"
                    cache = self._get_artifact_cache()
                    if artifact_key in cache:
                        metric_list = mlflow.artifacts.load_dict(f"{self.artifact_uri}/{artifact_key}")
                    else:
                        metric_list = []
                    metric_list.append(value)
                    mlflow.log_dict(metric_list, artifact_key, run_id=self.run_id)
                    self._invalidate_artifact_cache()
                except Exception as e:
                    print(f"[MLFlowLogger] Failed to log non-scalar metric '{key}' at step {step}: {e}")

        if scalar_metrics:
            try:
                mlflow.log_metrics(scalar_metrics, step=step, run_id=self.run_id)
            except Exception as e:
                print(f"[MLFlowLogger] Failed to log scalar metrics at step {step}: {e}")


    def log_params(self, params: dict) -> None:
        """
        Log hyperparameters and config values for the run.
        Should be called once at the start of training.

        Args:
            params: Flat mapping of parameter name to value.
        """
        self._assert_run_active()
        try:
            mlflow.log_params(params, run_id=self.run_id)
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log params: {e}")


    def log_best(self, metrics_dict: dict, epoch: int) -> None:
        """
        Persist the best observed metrics as a JSON artifact.

        Args:
            metrics_dict: Best metric values.
            epoch:        Epoch at which the best was achieved.
        """
        self._assert_run_active()
        best_metrics = {"epoch": epoch, **metrics_dict}
        try:
            mlflow.log_dict(best_metrics, "best_metrics.json", run_id=self.run_id)
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log best metrics: {e}")


    def log_image(self, image, image_name: str) -> None:
        """
        Log an image artifact under the 'images/' folder.

        Args:
            image_name: File name (e.g. 'confusion_matrix.png').
            image:      Image object accepted by mlflow.log_image.
        """
        self._assert_run_active()
        try:
            mlflow.log_image(image, artifact_file=f"images/{image_name}")
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log image '{image_name}': {e}")


    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def log_checkpoint(self, model, model_name: str, save_model_state: bool = False) -> None:
        """
        Save a model checkpoint to MLFlow.

        Args:
            model:            Model wrapper exposing .model, .optimizer, and .epoch.
            model_name:       Artifact sub-path for the checkpoint.
            save_model_state: If True, saves full training state (epoch + optimizer).
                              If False, saves only the model weights.
        """
        self._assert_run_active()
        if save_model_state:
            self._save_model_state(model, model_name)
        else:
            self._save_weights(model, model_name)


    def _save_model_state(self, model, model_name: str) -> None:
        """Save full training state dict (model weights + optimizer state + epoch)."""
        state = {
            "epoch": model.epoch,
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        }
        try:
            mlflow.pytorch.log_state_dict(state, f"checkpoints/{model_name}", run_id=self.run_id)
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log model state for '{model_name}': {e}")


    def _save_weights(self, model, model_name: str) -> None:
        """Save only the model weights (no optimizer or epoch state)."""
        try:
            mlflow.pytorch.log_model(model.model, f"checkpoints/{model_name}", run_id=self.run_id)
        except Exception as e:
            print(f"[MLFlowLogger] Failed to log model weights for '{model_name}': {e}")


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

    def _assert_run_active(self) -> None:
        if self.run_id is None or self._active_run is None:
            raise RuntimeError(
                "MLFlowLogger has no active run. "
                "Ensure the logger was properly initialized and close() has not been called."
            )


    def __repr__(self) -> str:
        status = "active" if self._active_run is not None else "closed"
        return (
            f"MLFlowLogger("
            f"experiment='{self.experiment_name}', "
            f"run_id='{self.run_id}', "
            f"status={status}, "
            f"tracking_uri='{self.tracking_uri}')"
        )