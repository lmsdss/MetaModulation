import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from cbn import CBN


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Conv_Standard(nn.Module):
    # 3     32       32
    def __init__(self, args, x_dim, hid_dim, z_dim):
        super(Conv_Standard, self).__init__()
        self.args = args
        # Four convolutional blocks and a classifier layer.
        self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                 self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim), Flatten())
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

    def conv_block(self, in_channels, out_channels):
        # Each convolutional block includes a convolutional layer,a batch normalization layer and a ReLU activation layer.
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):

        return self.net(x)

    def functional_conv_block_bn(self, x_1, x_2, weights, biases):
        x_1 = F.conv2d(x_1, weights, biases, padding=1)

        task2_size = x_2.shape[0] * x_2.shape[1]

        bn = CBN(task2_size, x_1.shape[1], x_1.shape[1], x_1.shape[0], x_1.shape[1], x_1.shape[2],
                 x_1.shape[3])

        x, cbn_kl_loss = bn(x_1, torch.flatten(x_2.mean(dim=-1).mean(dim=-1)))

        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        return x, cbn_kl_loss

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):

        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def functional_forward_bn(self, x, hidden_support_2, weights, is_training, is_bn_mix=True):

        global cbn_kl_loss
        if is_bn_mix:
            for block in range(0, 4):
                hidden_support_2 = self.functional_forward(hidden_support_2, weights, block, is_training)
                x, cbn_kl_loss = self.functional_conv_block_bn(x, hidden_support_2, weights[f'net.{block}.0.weight'],
                                                               weights[f'net.{block}.0.bias'])

            x = x.view(x.size(0), -1)

        else:
            for block in range(0, 4):
                x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                               weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                               is_training)

        return x, cbn_kl_loss

    def functional_forward(self, x, weights, sel_layer, is_training):

        x = self.functional_conv_block(x, weights[f'net.{sel_layer}.0.weight'], weights[f'net.{sel_layer}.0.bias'],
                                       weights.get(f'net.{sel_layer}.1.weight'), weights.get(f'net.{sel_layer}.1.bias'),
                                       is_training)

        return x
