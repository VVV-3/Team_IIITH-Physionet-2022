import torch.nn as nn
import torch
import numpy as np


def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    """

    def __init__(self, input_dim, blocks_dim, kernel_size,n_classes=3,dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        # n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        # n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        # downsample = _downsample(n_samples_in, n_samples_out)
        # padding = _padding(downsample, kernel_size[0])
        # self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size[0], bias=False,
        #                        stride=downsample, padding=padding)
        # self.bn1 = nn.BatchNorm1d(n_filters_out)
        
        n_filters_out, n_samples_out = input_dim[0], input_dim[1]

        # Residual block layers
        self.res_blocks_0 = []
        self.res_blocks_1 = []
        self.res_blocks_2 = []
        self.res_blocks_3 = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            # if i==0:
            #     continue
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            # print(n_filters_in, int(n_filters_out/4), downsample, kernel_size[0], dropout_rate)
            resblk1d = ResBlock1d(n_filters_in, int(n_filters_out/4), downsample, kernel_size[0], dropout_rate)
            self.add_module('resblock1d_0_{0}'.format(i), resblk1d)
            self.res_blocks_0 += [resblk1d]
            
            resblk1d = ResBlock1d(n_filters_in, int(n_filters_out/4), downsample, kernel_size[1], dropout_rate)
            self.add_module('resblock1d_1_{0}'.format(i), resblk1d)
            self.res_blocks_1 += [resblk1d]
            
            resblk1d = ResBlock1d(n_filters_in, int(n_filters_out/4), downsample, kernel_size[2], dropout_rate)
            self.add_module('resblock1d_2_{0}'.format(i), resblk1d)
            self.res_blocks_2 += [resblk1d]
            
            resblk1d = ResBlock1d(n_filters_in, int(n_filters_out/4), downsample, kernel_size[3], dropout_rate)
            self.add_module('resblock1d_3_{0}'.format(i), resblk1d)
            self.res_blocks_3 += [resblk1d]
            
            
        self.lin1 = nn.Linear(blocks_dim[-1][0]*blocks_dim[-1][1], int((blocks_dim[-1][0]*blocks_dim[-1][1])/2))
        self.relu1 = nn.ReLU(inplace=False)
        self.lin2 = nn.Linear(int((blocks_dim[-1][0]*blocks_dim[-1][1])/2), n_classes)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lin3 = nn.Linear(n_classes, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        # x = self.conv1(x)
        # x = self.bn1(x)

        # Residual blocks
        y = x
        for i in range(len(self.res_blocks_0)):
            x_0, y_0 = self.res_blocks_0[i](x, y)
            x_1, y_1 = self.res_blocks_1[i](x, y)
            x_2, y_2 = self.res_blocks_2[i](x, y)
            x_3, y_3 = self.res_blocks_3[i](x, y)
            
            x = torch.cat((x_0,x_1,x_2,x_3), 1)
            y = torch.cat((y_0,y_1,y_2,y_3), 1)
            # x,y = x_2,y_2
            
         # Flatten array
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.dropout1(x)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        
        # x = self.lin3(x)
        # x = self.sigm(x)

        return x