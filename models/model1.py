import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEEGNet(nn.Module):
    """
    DeepEEGNet is a convolutional neural network designed for EEG signal processing,
    featuring deep layers and residual connections to enhance feature extraction capabilities.
    
    Parameters:
    -----------
    input_shape : tuple
        Expected input shape of the data, formatted as (batch, time, EEG channels, 1).
    num_classes : int
        The number of output classes or predictions the network should generate.
    """
    def __init__(self, input_shape, num_classes):
        super(DeepEEGNet, self).__init__()
        self.input_shape = input_shape
        T, C = input_shape[1], input_shape[2]  # Time and Channel dimensions
        
        # Layer 1: Temporal Convolution
        # Applies convolution across the time dimension with 16 filters.
        self.temp_conv1 = nn.Conv2d(1, 16, (5, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization to stabilize learning
        
        # Layer 2: Temporal Convolution with Residual
        # Second temporal convolution layer with 32 filters, using a residual connection.
        self.temp_conv2 = nn.Conv2d(16, 32, (5, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.res1 = nn.Conv2d(16, 32, (1, 1))  # Residual connection to match dimension
        
        # Layer 3: Spatial Depthwise Convolution
        # Depthwise convolution applied spatially for each temporal feature map separately.
        self.spatial_dw_conv = nn.Conv2d(32, 32 * 2, (1, C), groups=32)
        self.bn3 = nn.BatchNorm2d(32 * 2)
        
        # Adaptive Pooling
        # Pooling layer that adapts to the required output size, here in the temporal dimension.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        
        # Dense Layers
        # Fully connected layers that map the features to the final output classes.
        self.fc1 = nn.Linear(32 * 2 * T, 100)  # First dense layer
        self.fc2 = nn.Linear(100, num_classes)  # Output layer
        
        # Activation and Dropout
        self.activation = nn.ELU()  # ELU activation function used for non-linearity
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting

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
            The output tensor after processing through the network, with log probabilities for each class.
        """
        x = self.bn1(self.activation(self.temp_conv1(x)))  # Layer 1 with activation and normalization
        res = self.res1(x)  # Prepare the residual for layer 2
        x = self.bn2(self.activation(self.temp_conv2(x) + res))  # Layer 2 with residual connection
        x = self.bn3(self.activation(self.spatial_dw_conv(x)))  # Layer 3 depthwise convolution
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the output for dense layer
        x = self.dropout(self.activation(self.fc1(x)))  # Apply dropout after first dense layer
        x = self.fc2(x)  # Final output layer
        return F.log_softmax(x, dim=1)  # Apply log softmax to output for classification
