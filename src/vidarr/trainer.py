import math
import os
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

from vidarr.dali import dali_train_loader, dali_val_loader
from vidarr.utils import freeze_model, print_rank_0, timed

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def save_checkpoint(model, save_dir: str, filename: str = "final_model.pt"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    state_dict = model.state_dict()
    print_rank_0(f"Saving final checkpoint to {save_path}...")
    torch.save(state_dict, save_path)


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


@torch.compile
def load_criterion(criterion_type: str):
    if criterion_type == "bcewithlogits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError()
    return criterion


def load_metric(metric_type: str, device="cuda"):
    if metric_type == "binary":
        metric = BinaryAccuracy()
    else:
        raise NotImplementedError()
    return metric.to(device=device)


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


def _step(
    model, optimizer, criterion, lr_scheduler, inputs, labels, scaler, metric, is_train: bool,
):
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16 if scaler is not None else torch.bfloat16,
    ):
        pred = model(inputs)
        loss = criterion(pred, labels)

    if metric:
        metric.update(pred, labels)

    if is_train:
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
        torch.compiler.cudagraph_mark_step_begin()
        inputs = batch_data[0]["data"]  # Shape: [B, C, H, W]
        labels = batch_data[0]["label"].float()
        loss, times = timed(
            lambda: _step(
                model,
                optimizer,
                criterion,
                lr_scheduler,
                inputs,
                labels,
                scaler,
                metric,
                is_train=True,
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
        torch.compiler.cudagraph_mark_step_begin()
        inputs = batch_data[0]["data"]  # Shape: [B, C, H, W]
        labels = batch_data[0]["label"].float()
        loss, times = timed(
            lambda: _step(model, None, criterion, None, inputs, labels, scaler, metric, is_train=False)
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


def run_training(
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
    checkpoint_dir,
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
            with torch.no_grad():
                val_loss = val_epoch(
                    model,
                    val_dataloader,
                    criterion,
                    scaler,
                    metric,
                    prof,
                    epoch,
                )

    save_checkpoint(model=model, save_dir=checkpoint_dir)


def train(
    model_name: str,
    train_dir: str,
    val_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_classes: int = 1,
    drop_rate: Optional[float] = 0.1,
    drop_path_rate: Optional[float] = None,
    scheduler_type: str = "cosine",
    warmup_steps: float = 0.10,
    decay_steps: Optional[float] = 0.10,
    num_threads: int = 4,
    image_size: int = 224,
    image_crop: int = 224,
    use_scaler: bool = False,
    use_compile: bool = False,
    train_classification_head: bool = False,
    metric_type: str = "binary",
    criterion_type: str = "bcewithlogits",
    profiler_dir: str = "./log",
    checkpoint_dir: str = "./checkpoints",
):
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
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        train_classification_head=train_classification_head,
    )
    if use_compile:
        model = torch.compile(model)

    metric = load_metric(metric_type=metric_type)
    optimizer = load_optimizer(model=model, lr=learning_rate)
    criterion = load_criterion(criterion_type=criterion_type)
    lr_scheduler = load_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )

    if use_scaler:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    run_training(
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
        checkpoint_dir=checkpoint_dir,
    )
