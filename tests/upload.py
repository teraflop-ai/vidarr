import timm
import torch

from vidarr.utils import upload_to_hf

model_name = "timm/efficientvit_b1.r288_in1k"
num_classes = 2
checkpoint_path = "/home/henry/vidarr/checkpoints/final_model.pt"
device = "cuda"
repo_id = "TeraflopAI/compression-detection"

model = timm.create_model(model_name, num_classes=num_classes)
state = torch.load(checkpoint_path, weights_only=True, map_location=device)
model.load_state_dict(state)

model_cfg = {
    "architecture": "efficientvit_b1",
    "num_classes": 2,
    "num_features": 256,
    "global_pool": "avg",
    "pretrained_cfg": {
        "tag": "r288_in1k",
        "custom_load": False,
        "input_size": [3, 288, 288],
        "fixed_input_size": False,
        "interpolation": "bicubic",
        "crop_pct": 1.0,
        "crop_mode": "center",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 1000,
        "pool_size": [9, 9],
        "first_conv": "stem.in_conv.conv",
        "classifier": "head.classifier.4",
    },
    "architectures": ["timm/efficientvit_b1.r288_in1k"],
    "num_labels": 2,
    "id2label": {1: "no_artifacts", 0: "jpeg_artifacts"},
    "label2id": {"no_artifacts": 1, "jpeg_artifacts": 0},
    "model_type": "image-classification",
}

model_card = {
    "tags": ["image-classification", "timm", "transformers"],
    "pipeline_tag": "image-classification",
    "library_name": "timm",
    "license": "other",
}

upload_to_hf(
    model=model,
    model_cfg=model_cfg,
    repo_id=repo_id,
    model_card=model_card,
    private=False,
)
