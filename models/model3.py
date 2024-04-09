import torch.nn as nn
import torch.nn.functional as F

class model3(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate):
        super(model3, self).__init__()
        _, T, C, _ = input_shape

        # First convolutional layer: Expands channel dimension from T to 16, using a kernel size of (1, 5)
        self.conv1 = nn.Conv2d(T, 16, (1, 5), padding='same')
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization for the first layer

        # Second convolutional layer: Expands channel dimension from 16 to 32, using the same kernel size (1, 5)
        self.conv2 = nn.Conv2d(16, 32, (1, 5), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        # Residual connection to adjust channel dimensions from 16 to 32
        self.res1 = nn.Conv2d(16, 32, (1, 1))
        
        # Third convolutional layer: Doubles the number of channels to 64 using a (1, 1) kernel and group convolution
        self.conv3 = nn.Conv2d(32, 32 * 2, (1, 1), groups=32)
        self.bn3 = nn.BatchNorm2d(32 * 2)
        
        # Adaptive average pooling layer to reduce the spatial dimensions to 1x1 for each feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Two fully connected layers: the first one maps the pooled features to a 100-dimensional space
        self.fc1 = nn.Linear(32 * 2, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
        # Activation function ELU used across the network
        self.activation = nn.ELU()
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply first convolutional layer, batch normalization, and activation
        x = self.bn1(self.activation(self.conv1(x)))
        # Save output for residual connection
        res = self.res1(x)
        # Apply second convolutional layer, add residual, apply batch normalization, and activation
        x = self.bn2(self.activation(self.conv2(x) + res))
        # Apply third convolutional layer, batch normalization, and activation
        x = self.bn3(self.activation(self.conv3(x)))
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        # Flatten the output and apply the first fully connected layer, dropout, and activation
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        # Apply second fully connected layer
        x = self.fc2(x)
        # Return the log_softmax of the output
        return F.log_softmax(x, dim=1)
