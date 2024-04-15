import torch.nn as nn
import torch.nn.functional as F

class model3(nn.Module):
    """
    A convolutional neural network designed for handling multidimensional data with both simple and complex
    convolutions. This model uses deep convolutional layers with increasing channel size, batch normalization,
    ELU activations, and adaptive pooling. It includes residual connections to help with training deeper networks
    and uses dropout for regularization. The model is suitable for classification tasks requiring extraction of
    complex patterns from spatial and temporal data.

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
        The number of neurons in the first fully connected layer after pooling.
    hparam4 : tuple, default=(1,5)
        The kernel size for the first convolutional layer.
    hparam5 : tuple, default=(1,5)
        The kernel size for the second convolutional layer.
    hparam6 : tuple, default=(1,1)
        The kernel size for the third convolutional layer, used primarily for group convolutions.

    Example
    -------
    >>> import torch
    >>> inp_tensor = torch.rand([1, 64, 3, 1])  # Example input tensor with shape (batch_size, T, C, spatial_dim)
    >>> model = model3(input_shape=inp_tensor.shape, num_classes=10, dropout_rate=0.3)
    >>> output = model(inp_tensor)
    >>> print(output.shape)
    torch.Size([1, 10])
    """
    def __init__(self, input_shape, num_classes, dropout_rate, hparam1 = 16, hparam2 = 32, hparam3 = 100, hparam4 = (1,5), hparam5 = (1,5), hparam6 = (1,1)):
        super(model3, self).__init__()
        _, T, C, _ = input_shape

        # First convolutional layer: Expands channel dimension from T to 16, using a kernel size of (1, 5)
        self.conv1 = nn.Conv2d(T, hparam1, hparam4, padding='same')
        self.bn1 = nn.BatchNorm2d(hparam1)  # Batch normalization for the first layer

        # Second convolutional layer: Expands channel dimension from 16 to 32, using the same kernel size (1, 5)
        self.conv2 = nn.Conv2d(hparam1, hparam2, hparam5, padding='same')
        self.bn2 = nn.BatchNorm2d(hparam2)
        # Residual connection to adjust channel dimensions from 16 to 32
        self.res1 = nn.Conv2d(hparam1, hparam2, (1, 1))
        
        # Third convolutional layer: Doubles the number of channels to 64 using a (1, 1) kernel and group convolution
        self.conv3 = nn.Conv2d(hparam2, hparam2 * 2, hparam6, groups=hparam2)
        self.bn3 = nn.BatchNorm2d(hparam2 * 2)
        
        # Adaptive average pooling layer to reduce the spatial dimensions to 1x1 for each feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Two fully connected layers: the first one maps the pooled features to a 100-dimensional space
        self.fc1 = nn.Linear(hparam2 * 2, hparam3)
        self.fc2 = nn.Linear(hparam3, num_classes)
        
        # Activation function ELU used across the network
        self.activation = nn.ELU()
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
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