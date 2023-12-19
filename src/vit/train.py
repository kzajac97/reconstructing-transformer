import os
from typing import Any, Callable, Iterable, Tuple

import torch
import wandb
from sklearn import metrics
from sklearn.model_selection import train_test_split


def log_training_progress(model: torch.nn.Module, epoch: int, data_loader: Iterable) -> None:
    """
    Logs progress of the model training to W&B
    :note: requires Wto be called inside W&B run
    """
    logs = {}
    targets, predictions = predict(model, data_loader)

    logs["accuracy"] = metrics.accuracy_score(targets.cpu(), predictions.cpu().argmax(dim=-1))
    logs["f1_score"] = metrics.f1_score(
        targets.cpu(), predictions.cpu().argmax(dim=-1), zero_division=0, average="macro"
    )

    wandb.log(logs, step=epoch)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in neural model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def predict(model: torch.nn.Module, data_loader: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prediction loop for the given model and data loader
    :note: assumes running on CUDA enabled device
    """
    model = model.eval()
    model.cuda()
    targets, predictions = [], []

    for inputs, y_true in data_loader:
        with torch.no_grad():
            inputs = inputs.cuda()
            y_true = y_true.cuda()

            y_pred = model(inputs)

            predictions.append(y_pred)
            targets.append(y_true)

    return torch.cat(targets), torch.cat(predictions)


def evaluate(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1_score": metrics.f1_score(y_true, y_pred, zero_division=0, average="macro"),
    }


def train_test_split_data_loader(dataset, test_size: float = 0.1):
    train_indices, validation_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    validation_data_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=128, shuffle=False)

    return train_data_loader, validation_data_loader


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    dataset: Iterable,
    n_epochs: int,
    validation_size: float = 0.1,
) -> torch.nn.Module:
    """
    Train loop for the given torch model
    :note: assumes running on CUDA enabled device
    """
    train_data_loader, validation_data_loader = train_test_split_data_loader(dataset, test_size=validation_size)

    for epoch in range(n_epochs):
        for x, y in train_data_loader:
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            y_pred = model(x)

            loss_value = loss_function(y_pred, y)
            loss_value.backward()
            optimizer.step()

        log_training_progress(model, epoch, validation_data_loader)

    return model


def run(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    train_dataset: Iterable,
    test_dataset: Iterable,
    n_epochs: int,
    config: dict[str, Any],
    validation_size=0.1,
) -> None:
    """
    Run training and evaluation loop for the given model and data

    :param model: pytorch model to train
    :param optimizer: pytorch optimizer to use
    :param loss_function: pytorch loss function to use
    :param train_dataset: iterable dataset for training
    :param test_dataset: iterable dataset for testing
    :param n_epochs: number of epochs to train
    :param config: dictionary with configuration for W&B
    :param validation_size: size of the validation set
    """
    with wandb.init(project=config["project_name"], name=config["run_name"]):
        wandb.log(config)
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
        # run training and evaluation
        model = train(model, optimizer, loss_function, train_dataset, n_epochs, validation_size)
        target, predictions = predict(model, test_data_loader)
        # log metrics and model info to W&B
        wandb.log(evaluate(target.cpu(), predictions.cpu().argmax(dim=-1)))
        wandb.log({"n_trainable_parameters": count_trainable_parameters(model)})
        # save model to file and upload to W&B
        path = os.path.join(wandb.run.dir, "model.pt")
        torch.save(model.state_dict(), path)
        wandb.save(path)
