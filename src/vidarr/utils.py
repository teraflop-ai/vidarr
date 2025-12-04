import torch
import wandb


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


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


def initial_write(
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
