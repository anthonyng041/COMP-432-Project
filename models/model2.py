import torch.nn as nn
import torch.nn.functional as F

class model2(nn.Module):
    """
    A convolutional neural network model that emphasizes spatial and temporal feature extraction with separate
    convolutions for different dimensions. This model uses ReLU activations, batch normalization, adaptive pooling, 
    and multiple fully connected layers, making it suitable for tasks that involve complex spatial-temporal dynamics 
    in the data.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input tensor, expected to be (batch_size, T, C, spatial_dim), where T is the temporal
        dimension and C is the number of channels.
    num_classes : int
        The number of output classes for the classification task.
    dropout_rate : float
        The dropout probability used in the dropout layer to prevent overfitting.
    hparam1 : int, default=16
        The number of filters in the first convolutional layer.
    hparam2 : int, default=32
        The number of filters in the second convolutional layer.
    hparam3 : int, default=100
        The number of neurons in the first fully connected layer.
    hparam4 : tuple, default=(3,1)
        The kernel size for the first convolutional layer.
    hparam5 : tuple, default=(1,3)
        The kernel size for the second convolutional layer.

    Example
    -------
    >>> import torch
    >>> inp_tensor = torch.rand([1, 64, 3, 1])  # Example input tensor with shape (batch_size, T, C, spatial_dim)
    >>> model = model2(input_shape=inp_tensor.shape, num_classes=5, dropout_rate=0.3)
    >>> output = model(inp_tensor)
    >>> print(output.shape)
    torch.Size([1, 5])
    """
    def __init__(self, input_shape, num_classes, dropout_rate, hparam1 = 16, hparam2 = 32, hparam3 = 100, hparam4 = (3,1), hparam5 = (1,3)):
        super(model2, self).__init__()
        _, T, C, _ = input_shape

        # First convolutional layer with 16 filters, kernel size (3, 1), applying padding to maintain dimension
        self.conv1 = nn.Conv2d(T, hparam1, hparam4, padding='same')
        self.bn1 = nn.BatchNorm2d(hparam1)  # Batch normalization for the first conv layer
        self.relu1 = nn.ReLU()  # ReLU activation function

        # Second convolutional layer with 32 filters, kernel size (1, 3), also with padding to maintain dimension
        self.conv2 = nn.Conv2d(hparam1, hparam2, hparam5, padding='same')
        self.bn2 = nn.BatchNorm2d(hparam2)
        self.relu2 = nn.ReLU()

        # Adaptive average pooling to reduce each feature map to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to expand the features to 100 units
        self.fc1 = nn.Linear(hparam2, hparam3)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization
        self.fc2 = nn.Linear(hparam3, num_classes)  # Final fully connected layer to classify into `num_classes` categories

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        # Forward pass definitions applying layers sequentially
        x = self.relu1(self.bn1(self.conv1(x)))  # Apply conv1, then bn1, then relu1
        x = self.relu2(self.bn2(self.conv2(x)))  # Apply conv2, then bn2, then relu2
        x = self.adaptive_pool(x)  # Apply adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.dropout(self.fc1(x))  # Apply first fully connected layer and then dropout
        x = self.fc2(x)  # Apply second fully connected layer
        return F.log_softmax(x, dim=1)  # Use log_softmax to output probabilities
