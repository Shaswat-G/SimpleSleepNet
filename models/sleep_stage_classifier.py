import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import for the Mish activation
import logging


logger = logging.getLogger(__name__)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SleepStageClassifier(nn.Module):
    """
    A neural network module for classifying sleep stages.
    
    This model takes input features and classifies them into different sleep stages.
    
    Args:
        input_dim (int): The dimension of the input features. Default is 128.
        num_classes (int): The number of output classes. Default is 5.
        dropout_probs (float): A single dropout probability for the dropout layers. Default is 0.5.
    Raises:
        ValueError: If dropout_probs is not a float.
    
    Attributes:
        classifier (nn.Sequential): A sequential container of layers including linear, 
                                    batch normalization, Mish activation, and dropout layers.
    
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, input_dim: int = 128, num_classes: int = 5, dropout_probs: float = 0.5):
        super(SleepStageClassifier, self).__init__()
        
        # Ensure dropout_probs is a float
        if not isinstance(dropout_probs, float):
            raise ValueError("dropout_probs must be a float.")
        
        logger.info(f"Initializing SleepStageClassifier with input_dim={input_dim}, num_classes={num_classes}, dropout_probs={dropout_probs}")
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),    # First linear layer
            nn.BatchNorm1d(256),          # Batch normalization
            Mish(),                       # Replace ReLU with Mish activation
            nn.Dropout(p=dropout_probs),  # First dropout layer
            nn.Linear(256, 128),          # Second linear layer
            nn.BatchNorm1d(128),          # Batch normalization
            Mish(),                       # Replace ReLU with Mish activation
            nn.Dropout(p=dropout_probs),  # Second dropout layer
            nn.Linear(128, num_classes)   # Output layer
        )
        
        logger.debug("SleepStageClassifier model architecture created.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        logger.debug("Forward pass started.")
        x = self.classifier(x)
        logger.debug("Forward pass completed.")
        return x