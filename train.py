import numpy as np
import timm
import torch
import torch.nn as nn
from torch import optim
from torch.profiler import profile

from src.vidarr.dali import dali_train_loader
from src.vidarr.utils import timed

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


images_dir = ""


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


def load_optimizer(model, lr: float = 3e-5, fused: bool = True):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, fused=fused
    )
    return optimizer


def load_criterion():
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def train_step(inputs, labels):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        pred = model(inputs)
        loss = criterion(pred, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


def val_step():
    pass


model = load_model(model_name="timm/efficientvit_b2.r288_in1k")  # "tiny_vit_21m_224"
optimizer = load_optimizer(model=model)
criterion = load_criterion()
scaler = torch.amp.GradScaler()

train_step = torch.compile(
    train_step, fullgraph=False, backend="inductor", mode="max-autotune"
)

train_data = dali_train_loader(images_dir=images_dir, batch_size=256, num_threads=8)

num_epochs = 5

model.train()
for epoch in range(num_epochs):
    timed_steps = []
    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/tinyvit"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch_data in enumerate(train_data):
            inputs = batch_data[0]["data"]  # Shape: [B, C, H, W]
            labels = batch_data[0]["label"].float()
            loss, times = timed(lambda: train_step(inputs, labels))
            prof.step()
            timed_steps.append(times)
            if step % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    med = np.median(timed_steps)
    print(f"epoch: {epoch + 1}, median step time: {med}")
