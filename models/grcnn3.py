import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 64, 128, 64, 32, 32, 64, 128, 64, 32, 32]
kernel_sizes = [9, 4, 4, 4, 4, 9, 4, 4, 4, 4, 9]
strides = [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1]


class GRCNN3(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=4)
        self.bn1 = nn.BatchNorm2d(filter_sizes[0])

        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=1)
        self.bn2 = nn.BatchNorm2d(filter_sizes[1])

        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.bn3 = nn.BatchNorm2d(filter_sizes[2])

        self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=4)
        self.bn4 = nn.BatchNorm2d(filter_sizes[3])

        self.conv5 = nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=1)
        self.bn5 = nn.BatchNorm2d(filter_sizes[4])

        self.res1 = ResidualBlock(filter_sizes[4], filter_sizes[4])
        self.res2 = ResidualBlock(filter_sizes[4], filter_sizes[4])
        self.res3 = ResidualBlock(filter_sizes[4], filter_sizes[4])

        self.convt1 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=4)
        self.bn6 = nn.BatchNorm2d(filter_sizes[5])

        self.convt2 = nn.ConvTranspose2d(filter_sizes[5], filter_sizes[6], kernel_sizes[6], stride=strides[6], padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(filter_sizes[6])

        self.convt3 = nn.ConvTranspose2d(filter_sizes[6], filter_sizes[7], kernel_sizes[7], stride=strides[7], padding=4, output_padding=1)
        self.bn8 = nn.BatchNorm2d(filter_sizes[7])

        self.convt4 = nn.ConvTranspose2d(filter_sizes[7], filter_sizes[8], kernel_sizes[8], stride=strides[8], padding=2, output_padding=1)
        self.bn9 = nn.BatchNorm2d(filter_sizes[8])

        self.convt5 = nn.ConvTranspose2d(filter_sizes[8], filter_sizes[9], kernel_sizes[9], stride=strides[9], padding=2, output_padding=1)
        self.bn10 = nn.BatchNorm2d(filter_sizes[9])

        self.convt6 = nn.ConvTranspose2d(filter_sizes[9], filter_sizes[10], kernel_sizes[10], stride=strides[10], padding=4)

        self.pos_output = nn.Conv2d(filter_sizes[10], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[10], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[10], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[10], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = F.relu(self.bn6(self.convt1(x)))
        x = F.relu(self.bn7(self.convt2(x)))
        x = F.relu(self.bn8(self.convt3(x)))
        x = F.relu(self.bn9(self.convt4(x)))
        x = F.relu(self.bn10(self.convt5(x)))
        x = F.relu(self.convt6(x))
        # x = self.convt3(x)                    # Testing acc was 0 if not add Relu! 

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        # # MSE Loss
        # p_loss = F.mse_loss(pos_pred, y_pos)
        # cos_loss = F.mse_loss(cos_pred, y_cos)
        # sin_loss = F.mse_loss(sin_pred, y_sin)
        # width_loss = F.mse_loss(width_pred, y_width)

        # Smooth L1 Loss
        Smooth_L1loss = nn.SmoothL1Loss()
        p_loss = Smooth_L1loss(pos_pred, y_pos)
        cos_loss = Smooth_L1loss(cos_pred, y_cos)
        sin_loss = Smooth_L1loss(sin_pred, y_sin)
        width_loss = Smooth_L1loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in