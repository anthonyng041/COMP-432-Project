import torch.nn as nn
import torch.nn.functional as F

class model1(nn.Module):
    """
    This model includes multiple convolutional layers with depth-wise separable 
    convolutions, batch normalization, ELU activations, adaptive average pooling, and a fully connected output layer.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    num_classes : int
        The number of output classes for the classification task.
    dropout_rate : float
        The dropout probability used in the dropout layer for regularization.
    hparam1 : int
        The number of filters in the first convolutional layer.
    hparam2 : int
        The number of filters in the third convolutional layer.
    hparam3 : int
        The number of filters in the fourth convolutional layer.
    hparam4 : tuple
        The kernel size for the first and fourth convolutional layers.
    hparam5 : tuple
        The kernel size for the third convolutional layer.

    Example
    -------
    >>> import torch
    >>> inp_tensor = torch.rand([1, 64, 3, 1])
    >>> model = model1(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([1, 10])
    """
    def __init__(self, input_shape, num_classes, dropout_rate, hparam1 = 32, hparam2 = 64, hparam3 = 128, hparam4 = (1,3), hparam5 = (1,1)):
        super(model1, self).__init__()
        # Extract T and C from input_shape
        _, T, C, _ = input_shape

        # First convolution layer
        self.conv1 = nn.Conv2d(T, hparam1, hparam4, padding='same')
        self.bn1 = nn.BatchNorm2d(hparam1)  # Batch normalization
        self.activation1 = nn.ELU()    # ELU activation function for non-linearity

        # Second convolution layer
        self.conv2 = nn.Conv2d(hparam1, hparam1, hparam4, groups=hparam1, padding='same')
        self.bn2 = nn.BatchNorm2d(hparam1)
        self.activation2 = nn.ELU()

        # Third convolution layer
        self.conv3 = nn.Conv2d(hparam1, hparam2, hparam5)
        self.bn3 = nn.BatchNorm2d(hparam2)
        self.activation3 = nn.ELU()

        # Fourth convolution layer
        self.conv4 = nn.Conv2d(hparam2, hparam3, hparam4, padding='same', dilation=2)
        self.bn4 = nn.BatchNorm2d(hparam3)
        self.activation4 = nn.ELU()

        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        # Fully connected layer to map the features to the class scores
        self.fc1 = nn.Linear(hparam3, num_classes)

        # Dropout for regularization to reduce overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        # Applying layers sequentially
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