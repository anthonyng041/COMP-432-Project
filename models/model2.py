import torch
import torch.nn as nn
import torch.nn.functional as F

class model2(torch.nn.Module):
    """EEGConvLSTMNet.
    
    A hybrid convolutional and LSTM network designed for decoding EEG signals.
    It utilizes convolutional layers to capture spatial features and LSTM layers to 
    capture temporal dependencies. The model is designed to process multi-channel EEG 
    time series data.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data (batch, time, EEG channel, channel).
    conv_channels : list of int
        Number of channels produced by the convolution operations for each layer.
    kernel_sizes : list of tuple
        Sizes of the kernels used in the convolutional layers.
    lstm_units : int
        Number of features in the hidden state h of the LSTM layer.
    dense_units : int
        Number of neurons in the fully connected layer that precedes the output.
    dropout_rate : float
        Dropout rate used for regularization to prevent overfitting.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 200, 32, 1])  # Batch size 1, 200 time points, 32 EEG channels
    >>> model = EEGConvLSTMNet(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    # Expected output: torch.Size([1, 4])
    """
    
    def __init__(
        self,
        input_shape,
        conv_channels=[16, 32],
        kernel_sizes=[(3, 1), (3, 1)],
        lstm_units=64,
        dense_units=4,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Dynamically add convolutional layers based on the conv_channels and kernel_sizes lists
        in_channels = 1
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
            )
            # Add max pooling layer to reduce the dimensionality of the feature maps
            self.pools.append(nn.MaxPool2d((2, 1)))
            # Add batch normalization to stabilize learning and accelerate training
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # LSTM layer to model temporal features. 
        # Using bidirectional LSTM to capture patterns both forwards and backwards in time.
        self.lstm = nn.LSTM(
            input_size=input_shape[2] * in_channels // 2 ** len(conv_channels),
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layer for output
        self.fc = nn.Linear(2 * lstm_units, dense_units)  # Bidirectional LSTM doubles the feature size

    def forward(self, x):
        # Pass the input through each convolutional block
        for conv, pool, norm in zip(self.convs, self.pools, self.batch_norms):
            x = F.relu(norm(pool(conv(x))))
        
        # Prepare the tensor for the LSTM layer
        x = x.permute(0, 2, 1, 3).contiguous()  # Reorder dimensions for LSTM compatibility
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the spatial dimensions for the LSTM

        # Apply LSTM layer
        x, _ = self.lstm(x)
        # Apply dropout after LSTM
        x = self.dropout(x[:, -1, :])  # Take only the output of the last sequence step
        # Apply fully connected layer to get the final output
        x = self.fc(x)

        # Use log softmax as the activation function for the output layer
        return F.log_softmax(x, dim=1)