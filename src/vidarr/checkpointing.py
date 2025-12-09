import os

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from vidarr.utils import print_rank_0


class Checkpoint(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict:
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


def save_checkpoint(model, save_dir: str, filename: str = "final_model.pt"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    state_dict = model.state_dict()
    print_rank_0(f"Saving final checkpoint to {save_path}...")
    torch.save(state_dict, save_path)
