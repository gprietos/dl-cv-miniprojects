import argparse
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from models.base import Model
from models.backbones import resnet18
from models.heads.classification_head import ClassificationHead
from data.dataloaders import get_dataloaders
from core.utils import load_config_dict
from core.loggers.mlflow_logger import MLFlowLogger
from core.metrics.utils import prepare_log_payload
from core.metrics.builder import build_classification_metrics
from core.checkpoint_manager import CheckpointManager


def train(config: dict) -> None:
    #! log grid ???
    # train_loader, val_loader, test_loader = get_dataloaders("/home/gprietos/AI/Datasets/image_classification/mini-imagenet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DataLoaders ---
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config["dataset"]["dataset_path"],
        batch_size=config["train"]["batch_size"]
    )

    classes = train_loader.dataset.classes
    num_classes = len(classes)
    
    # --- Model ---
    backbone = resnet18()
    head = ClassificationHead(in_features=backbone.out_features, num_classes=num_classes)
    model = Model(backbone, head)
    model = model.to(device)

    if config["model"]["compile_model"]:
        model.compile()

    # -- Loss Function ---
    loss_fn = nn.CrossEntropyLoss()

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["train"]["base_lr"],
                                  weight_decay=config["train"]["weight_decay"])


    # from core.optimizers.muon import SingleDeviceMuonWithAuxAdam
    # muon_params = []
    # adam_params = []

    # for name, p in model.named_parameters():
    #     if not p.requires_grad:
    #         continue

    #     if p.ndim == 2:
    #         # Linear layers only
    #         muon_params.append(p)
    #     else:
    #         # Conv kernels, biases, BN/LN params
    #         adam_params.append(p)

    # param_groups = [
    #     dict(
    #         params=muon_params,
    #         use_muon=True,
    #         lr=0.02,
    #         weight_decay=0.01,
    #     ),
    #     dict(
    #         params=adam_params,
    #         use_muon=False,
    #         lr=3e-4,
    #         betas=(0.9, 0.95),
    #         weight_decay=0.01,
    #     ),
    # ]

    # optimizer = SingleDeviceMuonWithAuxAdam(param_groups)




    # --- Metrics ---
    train_metrics = build_classification_metrics(num_classes=num_classes, per_class_metrics=True).to(device)
    val_metrics = build_classification_metrics(num_classes=num_classes, per_class_metrics=True).to(device)


    # --- AMP ---
    bf16 = True if config["model"]["dtype"] == "bfloat16" else False
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = GradScaler(enabled=not bf16)


    # --- Logger ---
    # logger = MLFlowLogger(
    #     experiment_name=config["logging"]["experiment_name"],
    #     run_name=f"resnet18_baseline_{config['train']['epochs']}epochs",
    #     tracking_uri=config["logging"]["tracking_uri"]
    # )
    logger = MLFlowLogger(
        experiment_name="resnet18_baseline",
        run_name=f"test",
        tracking_uri="http://localhost:5000/"
    )



    # ---- Checkpoint Manager ---
    checkpoint_manager = CheckpointManager(
        loggers=[logger],
        monitored_metric="val/f1/macro",
        mode="max",
        every_n_epochs=1,
    )



    accumulation_steps = config["train"]["grad_accum_steps"]

    #! Change for resume
    epoch_start = 0
    print(f"Starting training for {config['train']['epochs']} epochs with batch size {config['train']['batch_size']} and gradient accumulation steps {accumulation_steps}.") #!borrar

    for epoch in range(epoch_start, config["train"]["epochs"]):

        # --- Training ---
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        train_metrics.reset()

        for i, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=dtype):
                output = model(data)
                loss = loss_fn(output, target)

                loss /= accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                if config["train"]["clip_grad"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["train"]['max_grad_norm'])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
            total_train_loss += loss.item() * accumulation_steps

            with torch.no_grad():
                train_metrics.update(output, target)

        avg_train_loss = total_train_loss / len(train_loader)

        computed_train_metrics = train_metrics.compute()
        train_payload = prepare_log_payload(
            stage="train", 
            loss_value=avg_train_loss, 
            computed_metrics=computed_train_metrics, 
            class_names=classes
        )
        logger.log(train_payload, step=epoch)
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Loss: {avg_train_loss:.4f}") #!borrar


        # --- Validation ---
        model.eval()
        total_val_loss = 0.0

        val_metrics.reset()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(data)
                loss = loss_fn(output, target)
                total_val_loss += loss.item()

                val_metrics.update(output, target)

        avg_val_loss = total_val_loss / len(val_loader)

        computed_val_metrics = val_metrics.compute()
        val_payload = prepare_log_payload(
            stage="val",
            loss_value=avg_val_loss,
            computed_metrics=computed_val_metrics,
            class_names=classes
        )
        logger.log(val_payload, step=epoch)
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Validation Loss: {avg_val_loss:.4f}")


        # --- Checkpoint Management ---
        checkpoint_manager.save_periodic_checkpoint(
        epoch=epoch, model=model, optimizer=optimizer, scaler=scaler
        )

        checkpoint_manager.evaluate_best_model(
            epoch=epoch, metrics_dict=val_payload, model=model
        )
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ResNet-18 Baseline Training Script")
    parser.add_argument("--config_path", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_config_dict(args.config_path)
    train(config)


