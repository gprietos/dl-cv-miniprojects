from typing import List
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassPrecision, 
    MulticlassRecall, 
    MulticlassF1Score
)


def build_classification_metrics(
    num_classes: int,
    average: str = "macro",
    per_class_metrics: bool = False
) -> MetricCollection:
    """
    Fast online tracking metrics used across every batch.
    Returns a unified MetricCollection.
    """

    metrics_dict = {
        f"accuracy/{average}": MulticlassAccuracy(num_classes=num_classes, average=average),
        f"precision/{average}": MulticlassPrecision(num_classes=num_classes, average=average),
        f"recall/{average}": MulticlassRecall(num_classes=num_classes, average=average),
        f"f1/{average}": MulticlassF1Score(num_classes=num_classes, average=average),
    }
    
    # per-class tracking
    if per_class_metrics:
        metrics_dict["accuracy/per_class"] = MulticlassAccuracy(num_classes=num_classes, average="none")
        metrics_dict["precision/per_class"] = MulticlassPrecision(num_classes=num_classes, average="none")
        metrics_dict["recall/per_class"] = MulticlassRecall(num_classes=num_classes, average="none")
        metrics_dict["f1/per_class"] = MulticlassF1Score(num_classes=num_classes, average="none")
        
    return MetricCollection(metrics_dict)