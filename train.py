import math

import numpy as np
import timm
import torch
import torch.nn as nn
from torch import optim
from torch.profiler import profile
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from transformers import get_wsd_schedule

from src.vidarr.dali import dali_train_loader
from src.vidarr.utils import freeze_model, timed

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


images_dir = "/image_datasets/jpeg_experiment"
num_epochs = 20
batch_size = 256
learning_rate = 3e-5
warmup_steps = 0.10
decay_steps = 0.10


def load_model(
    model_name: str,
    num_classes: int = 1,
    device: str = "cuda",
):
    device = torch.device(device=device)
    model = timm.create_model(
        model_name=model_name, pretrained=True, num_classes=num_classes
    ).to(device)
    print(model)
    return model


def load_optimizer(model, lr: float, fused: bool = True):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, fused=fused
    )
    return optimizer


def load_criterion():
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def load_lr_scheduler(warmup_steps: float, decay_steps: float):
    num_warmup_steps = int(total_training_steps * warmup_steps)
    num_decay_steps = int(total_training_steps * decay_steps)
    num_train_steps = total_training_steps - num_warmup_steps - num_decay_steps

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        num_decay_steps=num_decay_steps,
    )
    return lr_scheduler


def train_step(inputs, labels):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16 if scaler is not None else torch.bfloat16,
    ):
        pred = model(inputs)
        loss = criterion(pred, labels)

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


def val_step():
    pass


train_data = dali_train_loader(
    images_dir=images_dir, batch_size=batch_size, num_threads=8
)
steps_per_epoch = math.ceil(train_data._size / batch_size)
total_training_steps = num_epochs * steps_per_epoch

model = load_model(
    model_name="tiny_vit_21m_224"
)  # "tiny_vit_21m_224" "timm/efficientvit_b2.r288_in1k"
freeze_model(model=model)

optimizer = load_optimizer(model=model, lr=learning_rate)
criterion = load_criterion()
lr_scheduler = load_lr_scheduler(warmup_steps=warmup_steps, decay_steps=decay_steps)
scaler = torch.amp.GradScaler()
metric = BinaryAccuracy()

train_step = torch.compile(
    train_step, fullgraph=False, backend="inductor", mode="max-autotune"
)

model.train()

with profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/tinyvit"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for epoch in range(num_epochs):
        metric.reset()
        timed_steps = []
        epoch_loss = 0
        pbar = tqdm(total=len(train_data), desc="Training")
        for step, batch_data in enumerate(train_data):
            inputs = batch_data[0]["data"]  # Shape: [B, C, H, W]
            labels = batch_data[0]["label"].float()
            loss, times = timed(lambda: train_step(inputs, labels))
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
