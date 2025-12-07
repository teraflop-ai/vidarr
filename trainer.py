import math
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from torch import optim
from torch.profiler import profile
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_wsd_schedule

from src.vidarr.dali import dali_train_loader, dali_val_loader
from src.vidarr.utils import freeze_model, print_rank_0, timed

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(
    model_name: str,
    num_classes: int = 1,
    drop_rate: float = 0.1,
    drop_path_rate: Optional[float] = 0.05,
    train_classification_head: bool = False,
    device: str = "cuda",
):
    device = torch.device(device=device)
    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    ).to(device)
    print_rank_0("=" * 80)
    print_rank_0(model)
    print_rank_0("=" * 80)

    if train_classification_head:
        model = freeze_model(model=model)
        print_rank_0("=" * 80)
    return model


def load_optimizer(model, lr: float, fused: bool = True):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, fused=fused
    )
    return optimizer


def load_criterion():
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def load_metric():
    metric = BinaryAccuracy()
    return metric


def load_lr_scheduler(
    optimizer,
    scheduler_type: str,
    total_training_steps: int,
    warmup_steps: float,
    decay_steps: Optional[float] = None,
):
    num_warmup_steps = int(total_training_steps * warmup_steps)

    if scheduler_type == "wsd":
        num_decay_steps = int(total_training_steps * decay_steps)
        num_train_steps = total_training_steps - num_warmup_steps - num_decay_steps
        lr_scheduler = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_decay_steps=num_decay_steps,
        )
    elif scheduler_type == "cosine":
        num_train_steps = total_training_steps - num_warmup_steps
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    else:
        raise NotImplementedError()
    return lr_scheduler


def train_step(
    model, optimizer, criterion, lr_scheduler, inputs, labels, scaler, metric
):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16 if scaler is not None else torch.bfloat16,
    ):
        pred = model(inputs)
        loss = criterion(pred, labels)

    if metric:
        metric.update(pred.detach().cpu(), labels.detach().cpu())

    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        old_scaler = scaler.get_scale()
        scaler.update()
        new_scaler = scaler.get_scale()
        if new_scaler >= old_scaler:
            lr_scheduler.step()
    else:
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    return loss


def val_step(model, criterion, inputs, labels, scaler, metric):
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16 if scaler is not None else torch.bfloat16,
        ):
            pred = model(inputs)
            loss = criterion(pred, labels)

    if metric:
        metric.update(pred.detach().cpu(), labels.detach().cpu())
    return loss


def train_epoch(
    model,
    train_data,
    optimizer,
    criterion,
    lr_scheduler,
    scaler,
    metric,
    prof,
    epoch,
):
    model.train()
    metric.reset()
    timed_steps = []
    epoch_loss = 0
    pbar = tqdm(total=len(train_data), desc="Training")
    for step, batch_data in enumerate(train_data):
        inputs = batch_data[0]["data"]  # Shape: [B, C, H, W]
        labels = batch_data[0]["label"].float()
        loss, times = timed(
            lambda: train_step(
                model,
                optimizer,
                criterion,
                lr_scheduler,
                inputs,
                labels,
                scaler,
                metric,
            )
        )
        prof.step()
        timed_steps.append(times)
        epoch_loss += loss.item()
        pbar.set_postfix(
            {
                "epoch": (epoch + 1),
                "accuracy": metric.compute().item(),
                "avg_loss": epoch_loss / (step + 1),
                "lr": lr_scheduler.get_last_lr()[0],
                "median step time": np.median(timed_steps),
            }
        )

        pbar.update()
    train_data.reset()
    pbar.close()
    return epoch_loss / (step + 1)


def val_epoch(model, val_data, criterion, scaler, metric, prof, epoch):
    model.eval()
    metric.reset()
    timed_steps = []
    epoch_loss = 0
    pbar = tqdm(total=len(val_data), desc="Evaluating")
    for step, batch_data in enumerate(val_data):
        inputs = batch_data[0]["data"]  # Shape: [B, C, H, W]
        labels = batch_data[0]["label"].float()
        loss, times = timed(
            lambda: val_step(model, criterion, inputs, labels, scaler, metric)
        )
        prof.step()
        timed_steps.append(times)
        epoch_loss += loss.item()
        pbar.set_postfix(
            {
                "epoch": (epoch + 1),
                "accuracy": metric.compute().item(),
                "avg_loss": epoch_loss / (step + 1),
                "median step time": np.median(timed_steps),
            }
        )

        pbar.update()
    val_data.reset()
    pbar.close()
    return epoch_loss / (step + 1)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    lr_scheduler,
    scaler,
    metric,
    num_epochs,
    profiler_dir,
):
    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=profiler_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            train_loss = train_epoch(
                model,
                train_dataloader,
                optimizer,
                criterion,
                lr_scheduler,
                scaler,
                metric,
                prof,
                epoch,
            )
            val_loss = val_epoch(
                model,
                val_dataloader,
                criterion,
                scaler,
                metric,
                prof,
                epoch,
            )


if __name__ == "__main__":
    model_name = "timm/efficientvit_m5.r224_in1k"
    train_dir = "/home/henry/Documents/image_datasets/jpeg_experiment/train_data"
    val_dir = "/home/henry/Documents/image_datasets/jpeg_experiment/val_data"
    num_epochs = 20
    batch_size = 1024
    learning_rate = 5.0e-05
    scheduler_type = "cosine"
    warmup_steps = 0.10
    decay_steps = 0.10
    num_threads = 8
    image_size = 224
    image_crop = 224
    use_scaler = False
    use_compile = False
    profiler_dir = "./log/tinyvit"

    train_dataloader = dali_train_loader(
        images_dir=train_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        image_size=image_size,
        image_crop=image_crop,
    )

    val_dataloader = dali_val_loader(
        images_dir=val_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        image_size=image_size,
        image_crop=image_crop,
    )

    steps_per_epoch = math.ceil(train_dataloader._size / batch_size)
    total_training_steps = num_epochs * steps_per_epoch

    model = load_model(
        model_name=model_name,
        drop_path_rate=None,
    )  # "tiny_vit_21m_224" "timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k"
    if use_compile:
        model = torch.compile(
            model, fullgraph=False, backend="inductor", mode="max-autotune"
        )

    metric = load_metric()
    optimizer = load_optimizer(model=model, lr=learning_rate)
    criterion = load_criterion()
    lr_scheduler = load_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
    )

    if use_scaler:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        metric=metric,
        num_epochs=num_epochs,
        profiler_dir=profiler_dir,
    )
