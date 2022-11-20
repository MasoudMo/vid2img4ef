import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
from math import floor
import torch.nn.functional as F


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Custom3DConv(nn.Module):

    def __init__(self,
                 out_channels,
                 kernel_sizes,
                 pool_sizes,
                 cnn_dropout_p):

        super().__init__()

        n_conv_layers = len(out_channels)

        # Default list arguments
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [2]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Compute paddings to preserve temporal dim
        paddings = list()
        for kernel_size in kernel_sizes:
            paddings.append(floor((kernel_size - 1) / 2))

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(Conv3DResBlock(in_channels=1,
                                                  padding=paddings[0],
                                                  out_channels=out_channels[0],
                                                  kernel_size=kernel_sizes[0],
                                                  pool_size=pool_sizes[0],
                                                  cnn_dropout_p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(Conv3DResBlock(in_channels=out_channels[layer_num-1],
                                                      padding=paddings[layer_num],
                                                      out_channels=out_channels[layer_num],
                                                      kernel_size=kernel_sizes[layer_num],
                                                      pool_size=pool_sizes[layer_num],
                                                      cnn_dropout_p=cnn_dropout_p)))
        # Change to sequential
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 7, 7)),
                                       nn.ReLU(inplace=True))

    def forward(self, x):

        # CNN layers
        x = self.conv(x)

        # FC layer
        x = self.output_fc(x)

        return x.squeeze(2)


class Conv3DResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 padding,
                 out_channels,
                 kernel_size,
                 pool_size,
                 cnn_dropout_p):

        super().__init__()

        # 1x1 convolution to make the channels equal for the residual
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(padding, padding, padding))
        self.bn = nn.InstanceNorm3d(out_channels)
        self.pool = nn.AvgPool3d(kernel_size=(1, pool_size, pool_size))
        self.dropout = nn.Dropout3d(p=cnn_dropout_p)

    def forward(self, x):

        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = x + residual
        x = self.pool(x)
        x = F.elu(x)

        return self.dropout(x)


class Conv3DEncoder(nn.Module):

    def __init__(self, num_conv_filters, conv_dropout_p):

        # assert(n_blocks >= 0)
        super().__init__()

        self.model = Custom3DConv(out_channels=num_conv_filters,
                                  kernel_sizes=[3] * len(num_conv_filters),
                                  pool_sizes=[2] * len(num_conv_filters),
                                  cnn_dropout_p=conv_dropout_p)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)


class ConvTransposeDecoder(nn.Module):

    def __init__(self, input_channels, output_channels, num_upsampling_layers):

        # assert(n_blocks >= 0)
        super().__init__()

        model = list()

        for i in range(num_upsampling_layers-1):
            model += [nn.ConvTranspose2d(int(input_channels / 2**(i)),
                                         int(input_channels / 2**(i+1)),
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(256 / 2**(i+1))),
                      nn.ReLU(True)]

        # Add the last layer
        model += [nn.ConvTranspose2d(int(input_channels / 2 ** (num_upsampling_layers-1)),
                                     output_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     bias=True),
                  nn.InstanceNorm2d(output_channels),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_channels, num_conv_filters=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_conv_filters (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()

        self.net = [
            nn.Conv2d(input_channels, num_conv_filters, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_conv_filters, num_conv_filters * 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(num_conv_filters * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_conv_filters * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
