import torch


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
