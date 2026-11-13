import torch
from typing import Dict, Any, List

def prepare_log_payload(
        stage: str,
        loss_value: float, 
        computed_metrics: Dict[str, torch.Tensor],
        class_names: List[str]
    ) -> Dict[str, float]:
    """
    Combines the loss and metric values into a single flat dictionary, 
    automatically mapping per-class arrays to descriptive class strings.
    """
    payload = {f"{stage}/loss": loss_value}
    
    for metric_name, val in computed_metrics.items():
        # Handle per-class breakdown (1D tensor matching class counts)
        if val.dim() == 1 and len(val) == len(class_names):
            base_name = metric_name.replace("/per_class", "")
            for idx, name in enumerate(class_names):
                payload[f"{stage}/{base_name}/{name}"] = val[idx].item()
        # Handle individual scalar values
        elif val.numel() == 1:
            payload[f"{stage}/{metric_name}"] = val.item()
            
    return payload