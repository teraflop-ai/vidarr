import numpy as np
import timm
import torch
import torch.nn as nn
from torch import optim

from src.vidarr.dali import dali_train_loader
from src.vidarr.utils import timed

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

class CFG:
    images_dir = "/image_datasets/jpeg_experiment"


def load_model(
    model_name: str = "tiny_vit_21m_224",
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


model = load_model(model_name="tiny_vit_21m_224")
optimizer = load_optimizer(model=model)
criterion = load_criterion()
scaler = torch.amp.GradScaler()

train_step = torch.compile(
    train_step, fullgraph=False, backend="inductor", mode="max-autotune"
)

train_data = dali_train_loader(images_dir=CFG.images_dir, batch_size=256)

num_epochs = 3

model.train()
for epoch in range(num_epochs):
    timed_steps = []
    for step, data in enumerate(train_data):
        inputs = data[0]["data"]
        labels = data[0]["label"].float()
        loss, times = timed(lambda: train_step(inputs, labels))
        timed_steps.append(times)
        if step % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    med = np.median(timed_steps)
    print(f"epoch: {epoch + 1}, median step time: {med}")
