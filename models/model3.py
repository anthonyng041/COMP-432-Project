import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMNet(nn.Module):
    """
    CNNLSTMNet is a hybrid neural network for EEG signal decoding that combines convolutional layers 
    with LSTM layers to capture both spatial and temporal features.

    The model is structured as follows:
    1. Convolutional Layer: To extract spatial features from the EEG signals.
    2. LSTM Layer: To capture temporal dependencies in the signals.
    3. Fully Connected Layer: To classify the features extracted by the CNN and LSTM layers.
    
    Parameters
    ----------
    input_shape : tuple
        The shape of the input EEG data, typically (batch_size, time_steps, channels, 1).
    num_classes : int
        The number of output classes for the classification task.
    lstm_hidden_size : int
        The number of features in the hidden state h of the LSTM.
    lstm_num_layers : int
        Number of recurrent layers in the LSTM.
    
    Example
    -------
    >>> batch_size, time_steps, channels = 1, 200, 32
    >>> inp_tensor = torch.rand([batch_size, time_steps, channels, 1])  # Example input tensor
    >>> model = CNNLSTMNet(input_shape=inp_tensor.shape, num_classes=4, lstm_hidden_size=64, lstm_num_layers=1)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([batch_size, num_classes])  # The output should match the number of classes and batch size
    """

    def __init__(self, input_shape, num_classes, lstm_hidden_size=64, lstm_num_layers=1):
        super(CNNLSTMNet, self).__init__()
        self.batch_size, self.T, self.C, _ = input_shape
        
        # CNN layer parameters
        self.cnn_out_channels = 32  # Number of output channels after the CNN layer
        self.kernel_size = (1, 5)   # Convolution kernel size
        self.stride = (1, 1)        # Convolution stride

        # LSTM layer parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Convolutional layer to extract spatial features from the EEG signals
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=self.cnn_out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding='same'
        )
        self.bn = nn.BatchNorm2d(self.cnn_out_channels)
        
        # LSTM layer to capture temporal dependencies
        # The input size is the product of the number of channels and the output channels from the CNN
        self.lstm = nn.LSTM(
            input_size=self.C * self.cnn_out_channels, 
            hidden_size=self.lstm_hidden_size, 
            num_layers=self.lstm_num_layers, 
            batch_first=True
        )
        
        # Fully connected layer to classify the features
        self.fc = nn.Linear(self.lstm_hidden_size, num_classes)

    def forward(self, x):
        # Apply convolutional layer
        x = F.relu(self.bn(self.conv(x)))
        
        # Prepare the input for the LSTM layer
        x = x.permute(0, 2, 1, 3)  # Permute to get the time dimension next to the batch dimension
        x = x.reshape(self.batch_size, self.T, -1)  # Flatten spatial dimensions
        
        # Apply LSTM layer
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # We take only the last output of the sequence
        
        # Apply fully connected layer and output the class probabilities
        x = self.fc(lstm_out)
        return F.log_softmax(x, dim=1)

# Example usage
# batch_size, time_steps, channels = 1, 200, 32
# inp_tensor = torch.rand([batch_size, time_steps, channels, 1])
# model = CNNLSTMNet(input_shape=inp_tensor.shape, num_classes=4, lstm_hidden_size=64, lstm_num_layers=1)
# output = model(inp_tensor)
# print(output.shape)
