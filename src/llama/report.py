import torch
from torch import Tensor


def tensor_size_in_gb(t: Tensor) -> float:
    """Returns tensor size in gigabytes"""
    gb_in_bytes = 1024**3
    return t.nelement() * t.element_size() / gb_in_bytes


def count_model_parameters(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_gb(model: torch.nn.Module) -> float:
    """Returns the size of a model in gigabytes"""
    parameter_size = sum(tensor_size_in_gb(p) for p in model.parameters() if p.requires_grad)
    buffer_size = sum(tensor_size_in_gb(b) for b in model.buffers())
    return parameter_size + buffer_size


def dataset_size_in_gb(dataloader: torch.utils.data.DataLoader) -> float:
    """
    Returns the size of a torch dataloader in gigabytes
    Assumes dataloader contains dicts of tensor, which is tokenized language data
    """
    return sum(sum(tensor_size_in_gb(t) for t in batch.values()) for batch, _ in dataloader)
