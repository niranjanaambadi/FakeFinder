"""
Optuna Hyperparameter Optimization for FakeFinder

This module implements automated hyperparameter tuning using Optuna to find
optimal CNN configurations for AI-generated image detection.

Author: Niranjana
Project: FakeFinder - AI-Generated Image Detection
"""

import os
from typing import Callable, Dict, Tuple

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import FlexibleCNN


def create_dataset_splits(data_path: str) -> Tuple[ImageFolder, ImageFolder]:
    """
    Creates training and validation datasets from a directory structure.
    
    Expected directory structure:
        data_path/
        ├── train/
        │   ├── real/
        │   └── fake/
        └── test/
            ├── real/
            └── fake/
    
    Args:
        data_path: Root path to the dataset directory
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "test")
    
    train_dataset = ImageFolder(root=train_path)
    val_dataset = ImageFolder(root=val_path)
    
    return train_dataset, val_dataset


def get_data_loaders(
    transform: transforms.Compose,
    batch_size: int,
    data_path: str,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders with the specified transform and batch size.
    
    Args:
        transform: Torchvision transforms to apply
        batch_size: Batch size for training
        data_path: Root path to the dataset
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "test")
    
    train_dataset = ImageFolder(root=train_path, transform=transform)
    val_dataset = ImageFolder(root=val_path, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def design_search_space(trial: optuna.Trial) -> Dict:
    """
    Design the hyperparameter search space for FlexibleCNN optimization.
    
    Search Space:
        - n_layers: 1-3 convolutional blocks
        - n_filters: 8-64 filters per layer (step of 8)
        - kernel_sizes: 3 or 5 per layer
        - dropout_rate: 0.1-0.5
        - fc_size: 64-512 (step of 64)
        - learning_rate: 1e-4 to 1e-2 (log scale)
        - resolution: 16, 32, or 64 pixels
        - batch_size: 8 or 16
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
    
    Returns:
        Dictionary containing all suggested hyperparameters
    """
    # CNN Architecture Hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    n_filters = [
        trial.suggest_int(f"n_filters_layer{i}", 8, 64, step=8)
        for i in range(n_layers)
    ]
    
    kernel_sizes = [
        trial.suggest_int(f"kernel_size_layer{i}", 3, 5, step=2)
        for i in range(n_layers)
    ]
    
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 512, step=64)
    
    # Training Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    resolution = trial.suggest_categorical("resolution", [16, 32, 64])
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    
    return {
        "n_layers": n_layers,
        "n_filters": n_filters,
        "kernel_sizes": kernel_sizes,
        "dropout_rate": dropout_rate,
        "fc_size": fc_size,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "batch_size": batch_size,
    }


def training_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    n_epochs: int,
    silent: bool = False
) -> float:
    """
    Execute one training epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        loss_fn: Loss function
        device: Device to run training on
        epoch: Current epoch number
        n_epochs: Total number of epochs
        silent: If True, suppress progress output
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    iterator = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{n_epochs}",
        disable=silent
    )
    
    for batch_idx, (inputs, labels) in enumerate(iterator):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if not silent and (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            iterator.set_postfix(loss=f"{avg_loss:.4f}")
    
    return running_loss / len(train_loader)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    silent: bool = False
) -> float:
    """
    Evaluate model accuracy on validation set.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        silent: If True, suppress output
    
    Returns:
        Validation accuracy (0-1 scale)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    
    if not silent:
        print(f"Validation Accuracy: {accuracy:.2%}")
    
    return accuracy


def objective_function(
    trial: optuna.Trial,
    device: torch.device,
    dataset_path: str,
    n_epochs: int = 4,
    silent: bool = True
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    This function:
    1. Samples hyperparameters from the search space
    2. Creates and trains a FlexibleCNN model
    3. Returns validation accuracy for Optuna to maximize
    
    Args:
        trial: Optuna trial object
        device: Device to run training on
        dataset_path: Path to the dataset
        n_epochs: Number of training epochs
        silent: If True, suppress training output
    
    Returns:
        Validation accuracy (Optuna will maximize this)
    """
    # Get hyperparameters from search space
    params = design_search_space(trial)
    
    # Create transform with sampled resolution
    transform = transforms.Compose([
        transforms.Resize((params["resolution"], params["resolution"])),
        transforms.ToTensor(),
    ])
    
    # Create model
    model = FlexibleCNN(
        n_layers=params["n_layers"],
        n_filters=params["n_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout_rate=params["dropout_rate"],
        fc_size=params["fc_size"]
    ).to(device)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        transform, params["batch_size"], dataset_path
    )
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(n_epochs):
        training_epoch(
            model, train_loader, optimizer, loss_fn,
            device, epoch, n_epochs, silent=silent
        )
    
    # Evaluation
    accuracy = evaluate_model(model, val_loader, device, silent=silent)
    
    return accuracy


def run_optimization(
    dataset_path: str,
    n_trials: int = 25,
    n_epochs: int = 4,
    study_name: str = "fakefinder_optimization",
    storage: str = None
) -> optuna.Study:
    """
    Run the complete hyperparameter optimization study.
    
    Args:
        dataset_path: Path to the dataset
        n_trials: Number of optimization trials
        n_epochs: Training epochs per trial
        study_name: Name for the Optuna study
        storage: Optional SQLite storage path for persistence
    
    Returns:
        Completed Optuna study object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create or load study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
    else:
        study = optuna.create_study(direction="maximize")
    
    # Run optimization
    study.optimize(
        lambda trial: objective_function(
            trial, device, dataset_path, n_epochs, silent=True
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best trial accuracy: {study.best_trial.value:.2%}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    return study


def get_trainable_params(model: nn.Module) -> int:
    """
    Calculate total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total count of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for FakeFinder"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Number of optimization trials (default: 25)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="Training epochs per trial (default: 4)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="SQLite storage path (e.g., sqlite:///study.db)"
    )
    
    args = parser.parse_args()
    
    study = run_optimization(
        dataset_path=args.data_path,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        storage=args.storage
    )
