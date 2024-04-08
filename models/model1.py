import torch.nn as nn
import torch.nn.functional as F

class model1(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate):
        super(model1, self).__init__()
        # Assumes input_shape is in the form (_, T, C, _), where T and C are the temporal and channel dimensions respectively.
        _, T, C, _ = input_shape

        # First convolution layer with 32 filters, kernel size (1,3), padding enabled to keep dimensions constant
        self.conv1 = nn.Conv2d(T, 32, (1, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for stability
        self.activation1 = nn.ELU()    # ELU activation function for non-linearity

        # Second convolution layer, using grouped convolutions for depthwise convolutions
        self.conv2 = nn.Conv2d(32, 32, (1, 3), groups=32, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.activation2 = nn.ELU()

        # Third convolution layer increasing channels from 32 to 64
        self.conv3 = nn.Conv2d(32, 64, (1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.activation3 = nn.ELU()

        # Fourth convolution layer further increasing channels to 128, with dilation to increase the receptive field
        self.conv4 = nn.Conv2d(64, 128, (1, 3), padding='same', dilation=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.activation4 = nn.ELU()

        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to map the features to the class scores
        self.fc1 = nn.Linear(128, num_classes)

        # Dropout for regularization to reduce overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Applying layers sequentially with functional connectivity for better flow
        x = self.bn1(self.activation1(self.conv1(x)))
        x = self.bn2(self.activation2(self.conv2(x)))
        x = self.bn3(self.activation3(self.conv3(x)))
        x = self.bn4(self.activation4(self.conv4(x)))
        x = self.adaptive_pool(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        # Applying log_softmax for the output layer to prepare for a classification task
        return F.log_softmax(x, dim=1)