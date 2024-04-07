import torch
import torch.nn as nn

class CompactEEGNet(nn.Module):
    """
    CompactEEGNet is designed to be a lightweight and efficient neural network
    for processing EEG signals, incorporating grouped convolutions and attention mechanisms.
    
    Parameters:
    -----------
    input_shape : tuple
        Expected input shape of the data, formatted as (batch, time, EEG channels, 1).
    num_classes : int
        The number of output classes or predictions the network should generate.
    """
    def __init__(self, input_shape, num_classes):
        super(CompactEEGNet, self).__init__()
        self.input_shape = input_shape
        _, T, C, _ = input_shape
        
        # Layer 1: Grouped Temporal Convolution
        self.group_conv1 = nn.Conv2d(1, 32, (3, 1), groups=1, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.ReLU()  # ReLU for non-linearity
        
        # Attention Layer
        self.attention = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(3, 1), padding='same', bias=False),
            nn.Softmax(dim=2)  # Softmax over the time dimension
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense Layer (Output)
        self.fc = nn.Linear(32, num_classes)  # Fully connected layer

    def forward(self, x):
        """
        Defines the computation performed at every call of the model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape matching the input_shape.
        
        Returns:
        --------
        torch.Tensor
            The output tensor after processing through the network, with logits for each class.
        """
        x = self.bn1(self.activation(self.group_conv1(x)))  # Apply grouped convolution, activation, and normalization
        x = x * self.attention(x)  # Apply attention to enhance important features
        x = self.global_pool(x)  # Apply global average pooling
        x = torch.flatten(x, 1)  # Flatten the features
        x = self.fc(x)  # Output layer
        return x