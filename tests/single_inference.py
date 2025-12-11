from urllib.request import urlopen

import timm
import torch
from PIL import Image

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)

model = timm.create_model(
    "hf_hub:TeraflopAI/compression-detection-288", pretrained=True,
)
model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

tensor = transforms(img).unsqueeze(0)
with (
    torch.inference_mode(),
    torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16),
):
    output = model(tensor)

pred_scores = torch.argmax(output, dim=1).item()

print(pred_scores)
