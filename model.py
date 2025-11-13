"""
FlexibleCNN: A Dynamic Convolutional Neural Network for Image Classification

This module implements a flexible CNN architecture that can be configured with
varying numbers of layers, filter sizes, and kernel dimensions. Designed for
use with hyperparameter optimization frameworks like Optuna.

Author: Niranjana
Project: FakeFinder - AI-Generated Image Detection
"""

import torch
import torch.nn as nn


class FlexibleCNN(nn.Module):
    """
    A flexible CNN architecture that dynamically builds convolutional layers
    based on hyperparameters.
    
    The network consists of:
    - Variable number of convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
    - Dynamically calculated flattened features
    - Fully connected classifier with dropout regularization
    
    Args:
        n_layers (int): Number of convolutional blocks
        n_filters (list[int]): Number of filters for each conv layer
        kernel_sizes (list[int]): Kernel size for each conv layer
        dropout_rate (float): Dropout probability for regularization
        fc_size (int): Size of the hidden fully connected layer
        num_classes (int): Number of output classes (default: 2 for binary classification)
    
    Example:
        >>> model = FlexibleCNN(
        ...     n_layers=3,
        ...     n_filters=[32, 64, 128],
        ...     kernel_sizes=[3, 3, 3],
        ...     dropout_rate=0.3,
        ...     fc_size=256
        ... )
        >>> x = torch.randn(1, 3, 64, 64)  # Batch of RGB images
        >>> output = model(x)  # Shape: (1, 2)
    """
    
    def __init__(
        self,
        n_layers: int,
        n_filters: list,
        kernel_sizes: list,
        dropout_rate: float,
        fc_size: int,
        num_classes: int = 2
    ):
        super(FlexibleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.features = nn.ModuleList()
        in_channels = 3  # RGB input images
        
        # Build convolutional blocks dynamically
        for i in range(n_layers):
            out_channels = n_filters[i]
            kernel_size = kernel_sizes[i]
            padding = (kernel_size - 1) // 2  # Same padding
            
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.features.append(block)
            in_channels = out_channels
        
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size
        
        # Classifier will be initialized after calculating flattened size
        self.classifier = None
        self._flattened_size = None
    
    def _create_classifier(self, flattened_size: int):
        """
        Creates the fully connected classifier based on the flattened feature size.
        
        Args:
            flattened_size: Size of the flattened feature maps
        """
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc_size, self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Apply convolutional feature extraction layers
        for layer in self.features:
            x = layer(x)
        
        # Flatten feature maps
        x = torch.flatten(x, start_dim=1)
        
        # Create classifier dynamically on first forward pass
        if self.classifier is None:
            self._flattened_size = x.shape[1]
            self._create_classifier(self._flattened_size)
            self.classifier.to(x.device)
        
        # Classification
        return self.classifier(x)
    
    def get_trainable_params(self) -> int:
        """
        Calculate the total number of trainable parameters.
        
        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_params(params: dict) -> FlexibleCNN:
    """
    Factory function to create a FlexibleCNN from a parameter dictionary.
    
    Args:
        params: Dictionary containing model hyperparameters
            - n_layers: Number of convolutional layers
            - n_filters: List of filter counts per layer
            - kernel_sizes: List of kernel sizes per layer
            - dropout_rate: Dropout probability
            - fc_size: Fully connected layer size
    
    Returns:
        Configured FlexibleCNN instance
    """
    return FlexibleCNN(
        n_layers=params["n_layers"],
        n_filters=params["n_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout_rate=params["dropout_rate"],
        fc_size=params["fc_size"]
    )


if __name__ == "__main__":
    # Example usage and quick test
    model = FlexibleCNN(
        n_layers=3,
        n_filters=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        dropout_rate=0.5,
        fc_size=128
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"\nModel architecture:\n{model}")
