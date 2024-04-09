import torch.nn as nn
import torch.nn.functional as F

class model2(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate):
        super(model2, self).__init__()
        _, T, C, _ = input_shape

        # First convolutional layer with 16 filters, kernel size (3, 1), applying padding to maintain dimension
        self.conv1 = nn.Conv2d(T, 16, (3, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization for the first conv layer
        self.relu1 = nn.ReLU()  # ReLU activation function

        # Second convolutional layer with 32 filters, kernel size (1, 3), also with padding to maintain dimension
        self.conv2 = nn.Conv2d(16, 32, (1, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Adaptive average pooling to reduce each feature map to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to expand the features to 100 units
        self.fc1 = nn.Linear(32, 100)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization
        self.fc2 = nn.Linear(100, num_classes)  # Final fully connected layer to classify into `num_classes` categories

    def forward(self, x):
        # Forward pass definitions applying layers sequentially
        x = self.relu1(self.bn1(self.conv1(x)))  # Apply conv1, then bn1, then relu1
        x = self.relu2(self.bn2(self.conv2(x)))  # Apply conv2, then bn2, then relu2
        x = self.adaptive_pool(x)  # Apply adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.dropout(self.fc1(x))  # Apply first fully connected layer and then dropout
        x = self.fc2(x)  # Apply second fully connected layer
        return F.log_softmax(x, dim=1)  # Use log_softmax to output probabilities
