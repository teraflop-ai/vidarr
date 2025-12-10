from typing import Optional

import timm
import torch
import torch.nn as nn
import wandb


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("No Linear layers found in model.")

    last_linear = linear_layers[-1]

    for p in last_linear.parameters():
        p.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print_rank_0("Trainable parameters:", trainable)
    return model


def timed(fn):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    training_time_ms = start_event.elapsed_time(end_event)
    training_time_s = training_time_ms / 1000.0
    return result, training_time_s


def model_num_params(model):
    non_embedding_params = sum(
        param.numel() for name, param in model.named_parameters() if "embed" not in name
    )
    return non_embedding_params


def initialize_wanb(
    entity: str,
    project: str,
    global_batch_size: int,
    learning_rate: float,
    num_training_epochs: int,
    gradient_accumulation_steps: int,
    model_name: str,
    dataset_name: str,
):
    writer = wandb.init(
        entity=entity,
        project=project,
        name=f"BS: {global_batch_size} LR: {learning_rate}",
        config={
            "learning_rate": learning_rate,
            "model": model_name,
            "dataset": dataset_name,
            "epochs": num_training_epochs,
            "gradient accumulation steps": gradient_accumulation_steps,
        },
    )
    return writer


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            print(message, flush=True)
    elif torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def upload_to_hf(
    model,
    model_cfg,
    repo_id: str = "my-model",
    token: Optional[str] = None,
    private: bool = False,
):
    timm.models.push_to_hf_hub(
        model=model,
        repo_id=repo_id,
        token=token,
        private=private,
        model_config=model_cfg,
    )
