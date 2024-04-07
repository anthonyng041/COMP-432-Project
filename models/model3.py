import torch
import torch.nn as nn

class AdvancedEEGNet(nn.Module):
    """
    AdvancedEEGNet incorporates modern deep learning architectures to provide a comprehensive
    tool for EEG signal analysis, suitable for complex and high-accuracy requirements.
    
    Parameters:
    -----------
    input_shape : tuple
        Expected input shape of the data, formatted as (batch, time, EEG channels, 1).
    num_classes : int
        The number of output classes or predictions the network should generate.
    """
    def __init__(self, input_shape, num_classes):
        super(AdvancedEEGNet, self).__init__()
        _, T, C, _ = input_shape
        
        # Layer 1: Dilated Temporal Convolution
        self.dilated_conv = nn.Conv2d(1, 64, (3, 1), dilation=(2, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        
        # Residual Block
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 128, (3, 1), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, (1, 1)),  # Pointwise to match dimensions
            nn.BatchNorm2d(64)
        )
        self.relu = nn.ReLU()

        # Depthwise and Pointwise Convolution
        self.depthwise = nn.Conv2d(64, 64 * 2, (1, C), groups=64)
        self.pointwise = nn.Conv2d(64 * 2, 128, (1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.activation = nn.ELU()

        # Global Pooling and Output Layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

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
        x = self.bn1(self.relu(self.dilated_conv(x)))
        x = x.permute(0, 3, 2, 1)  # Rearrange dimensions for attention
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 3, 2, 1)  # Revert dimensions
        res = x
        x = self.res_block(x) + res  # Apply residual connection
        x = self.depthwise(x)
        x = self.bn2(self.activation(self.pointwise(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x