from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Module


def predict(model: Module, dataloader: Iterable, limit_batches: int | None = None) -> tuple[Tensor, Tensor]:
    """
    Simple inference running using a model and a dataloader. Returns both predictions and targets.
    Additionally, the number of batches to process can be limited by setting `limit_batches`.
    """
    outputs = []
    targets = []

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches and batch_idx >= limit_batches:
            break  # stop when limit reached

        with torch.no_grad():
            target = batch.pop("targets")
            pred = model(**batch)
            # store both prediction and target for given batch
            outputs.append(pred["logits"])
            targets.append(target)

    return torch.cat(outputs), torch.cat(targets)
