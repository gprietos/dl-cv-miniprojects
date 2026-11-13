"""
checkpointing/monitor.py
------------------------
Metric-tracking logic for checkpointing decisions.

Deliberately has zero knowledge of:
  - how checkpoints are written to disk  (→ saver.py)
  - how metrics are logged               (→ training loop / logger)
  - models, optimizers, or any nn.Module

The monitor is a pure decision-maker.  It accepts a metric value and
returns a CheckpointEvent describing what the caller should do.
The training loop owns the actual saving and logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class CheckpointAction(Flag):
    """
    Bit-flag so the caller can test what happened in a single expression.

    Examples::

        event = monitor.update(metrics, epoch)
        if CheckpointAction.SAVE_BEST in event.actions:
            ...
        if CheckpointAction.SAVE_PERIODIC in event.actions:
            ...
        if event.is_best:          # convenience shorthand
            ...
    """
    NONE           = 0
    SAVE_BEST      = auto()   # monitored metric improved
    SAVE_PERIODIC  = auto()   # every_n_epochs tick


@dataclass(frozen=True)
class CheckpointEvent:
    actions:            CheckpointAction
    metric_value:       float
    best_metric_value:  float
    epoch:              int

    @property
    def is_best(self) -> bool:
        return CheckpointAction.SAVE_BEST in self.actions

    @property
    def is_periodic(self) -> bool:
        return CheckpointAction.SAVE_PERIODIC in self.actions

    @property
    def should_save(self) -> bool:
        return self.actions != CheckpointAction.NONE


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class CheckpointMonitor:
    """
    Tracks the best value of a monitored metric and decides when to checkpoint.

    Usage::

        monitor = CheckpointMonitor(
            monitored_metric="valid/loss",
            mode="min",
            every_n_epochs=5,
        )

        # inside the training loop, after computing epoch metrics:
        event = monitor.update(metrics_dict, epoch)
        if event.is_best:
            saver.save_checkpoint("checkpoint_best.pt", ...)
            logger.log_best(metrics_dict, epoch)
        if event.is_periodic:
            saver.save_checkpoint(f"checkpoint_{epoch:03d}.pt", ...)

    Args:
        monitored_metric:   Key that must be present in the metrics dict
                            passed to `update()`.  Use the same namespaced
                            key as in your logging dict, e.g. "valid/loss".
        stage:              Informational only — which stage produces the
                            metric ("train" or "valid").  The training loop
                            is responsible for calling `update()` at the
                            right stage; the monitor does not enforce this.
        mode:               "min" or "max".
        every_n_epochs:     If > 0, also signals a periodic save every N
                            epochs (epoch is 1-indexed when checking).
                            0 disables periodic saving.
        initial_best:       Seed the best-value tracker (e.g. when resuming).
                            If None, uses ±inf depending on mode.
    """

    def __init__(
        self,
        monitored_metric: str,
        stage: str = "valid",
        mode: str = "min",
        every_n_epochs: int = 0,
        initial_best: Optional[float] = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        if every_n_epochs < 0:
            raise ValueError(f"every_n_epochs must be >= 0, got {every_n_epochs}.")
        if not monitored_metric:
            raise ValueError("monitored_metric must be a non-empty string.")

        self.monitored_metric = monitored_metric
        self.stage            = stage
        self.mode             = mode
        self.every_n_epochs   = every_n_epochs

        if initial_best is not None:
            self._best = initial_best
        else:
            self._best = float("-inf") if mode == "max" else float("inf")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def best_metric_value(self) -> float:
        return self._best

    def update(self, metrics_dict: dict, epoch: int) -> CheckpointEvent:
        """
        Evaluate the current epoch's metrics and return a CheckpointEvent.

        Args:
            metrics_dict:   Flat dict of metric name → scalar value.
                            Must contain `self.monitored_metric`.
            epoch:          Current epoch (1-indexed).

        Raises:
            KeyError:  if `monitored_metric` is absent from `metrics_dict`.
            TypeError: if the metric value is not a number.
        """
        if self.monitored_metric not in metrics_dict:
            raise KeyError(
                f"Monitored metric '{self.monitored_metric}' not found in metrics_dict. "
                f"Available keys: {list(metrics_dict.keys())}"
            )

        value = metrics_dict[self.monitored_metric]
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Metric '{self.monitored_metric}' must be a scalar (int/float), "
                f"got {type(value).__name__}."
            )

        actions = CheckpointAction.NONE

        if self._is_improvement(value):
            self._best = value
            actions |= CheckpointAction.SAVE_BEST

        if self.every_n_epochs > 0 and epoch % self.every_n_epochs == 0:
            actions |= CheckpointAction.SAVE_PERIODIC

        return CheckpointEvent(
            actions=actions,
            metric_value=value,
            best_metric_value=self._best,
            epoch=epoch,
        )

    def __repr__(self) -> str:
        return (
            f"CheckpointMonitor("
            f"metric='{self.monitored_metric}', "
            f"mode='{self.mode}', "
            f"best={self._best:.6g}, "
            f"every_n_epochs={self.every_n_epochs})"
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best
        return value < self._best