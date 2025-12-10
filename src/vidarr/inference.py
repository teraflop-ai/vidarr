import timm
import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from vidarr.dali import dali_val_loader


def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: str):
    model = timm.create_model(model_name, num_classes=num_classes)
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_metric(metric_type: str, num_classes: int, device: str):
    metric = Accuracy(task=metric_type, num_classes=num_classes).to(device)
    return metric


def test(
    model_name: str,
    checkpoint_path: str,
    test_dir: str,
    metric_type: str = "multiclass",
    num_classes: int = 2,
    batch_size: int = 32,
    num_threads: int = 4,
    image_size: int = 224,
    crop_size: int = 224,
    use_compile: bool = False,
    device: str = "cuda",
):
    device = torch.device(device=device)

    dataloader = dali_val_loader(
        images_dir=test_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        image_size=image_size,
        image_crop=crop_size,
    )

    metric = load_metric(
        metric_type=metric_type, num_classes=num_classes, device=device
    )

    model = load_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
    )
    if use_compile:
        model = torch.compile(model=model)

    pbar = tqdm(total=len(dataloader), desc="Testing")
    with torch.inference_mode():
        for batch_data in dataloader:
            inputs = batch_data[0]["data"]
            labels = batch_data[0]["label"].squeeze().long()
            logits = model(inputs)
            pred_scores = torch.argmax(logits, dim=1)
            metric.update(pred_scores, labels)
            pbar.update()

    accuracy = metric.compute().item() * 100
    dataloader.reset()
    pbar.close()
    print(f"Test Accuracy: {accuracy:.4f}%")
    return accuracy
