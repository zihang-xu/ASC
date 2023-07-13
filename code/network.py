from torch import nn
from torch import cat
import torch
import functools

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(conv_block, self).__init__()

        insert_channels = out_channels if in_channels > out_channels else out_channels // 2
        layers = [
            nn.Conv3d(in_channels, insert_channels, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(insert_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        # 是否插入正则化层
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(insert_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)  # conv
        out = self.pool(x)  # down
        return x, out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(Up, self).__init__()
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')  # 三次线性插值（trilinear）
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

        self.conv_block = conv_block(in_channels + in_channels//2, out_channels, batch_norm)

    def forward(self, x, conv):
        x = self.sample(x)  # up
        x = cat((x, conv), dim=1)  # skip connect
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_filters=16, class_num=3, batch_norm=True, sample=True, has_dropout=False):
        super(UNet, self).__init__()

        self.down1 = Down(in_channels, num_filters, batch_norm)
        self.down2 = Down(num_filters, num_filters * 2, batch_norm)
        self.down3 = Down(num_filters * 2, num_filters * 4, batch_norm)
        self.down4 = Down(num_filters * 4, num_filters * 8, batch_norm)

        self.bridge = conv_block(num_filters * 8, num_filters * 16, batch_norm)

        self.up1 = Up(num_filters * 16, num_filters * 8, batch_norm, sample)
        self.up2 = Up(num_filters * 8, num_filters * 4, batch_norm, sample)
        self.up3 = Up(num_filters * 4, num_filters * 2, batch_norm, sample)
        self.up4 = Up(num_filters * 2, num_filters, batch_norm, sample)

        self.conv_class = nn.Conv3d(num_filters, class_num, 1)

        # droppout rate = 0.5 用了两个dropout
        self.has_dropout = has_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        conv1, x = self.down1(x)
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        conv4, x = self.down4(x)
        x = self.bridge(x)
        # dropout
        if self.has_dropout:
            x = self.dropout(x)

        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)
        # dropout
        if self.has_dropout:
            x = self.dropout(x)
        out = self.conv_class(x)

        return out


class FCDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, ndf=32):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1)  # 72
        self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)  # 36
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)  # 18
        self.conv4 = nn.Conv3d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)  # 9
        self.classifier = nn.Linear(ndf*8, out_channels)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature, image=None):
        x = self.conv1(feature)
        if image is not None:
            image_feature = self.conv0(image)
            x = torch.add(x, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        conv_maps = maps
        x = maps.view(maps.size(0), -1)
        x = self.classifier(x)
        x = self.sigmoid(x)

        return {'output':x, 'feat':conv_maps}


class PixelGAN(nn.Module):
    def __init__(self, in_channels=1, ndf=32, classes=4):
        super(PixelGAN, self).__init__()
        self.ndf = ndf
        channels = [ndf, ndf*2, ndf*4, ndf*8, ndf*16]
        kernel_size = 1
        stride = 1
        padding = 0
        # state size = 160
        self.conv_bolck1 = self.__conv_block(in_channels, channels[0], kernel_size, stride, padding, "first_layer")#128
        self.conv_bolok2 = self.__conv_block(channels[0], channels[1], kernel_size, stride, padding)#64
        self.conv_bolok3 = self.__conv_block(channels[1], 1, kernel_size, stride, padding, "last_layer")#32

    def __conv_block(self, inchannel, outchannel, kernel_size, stride, padding, layer=None):
        if layer == "first_layer":
            conv = nn.Conv3d(inchannel, outchannel, kernel_size, stride, padding)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, leakrelu)
        elif layer == "last_layer":
            conv = nn.Conv3d(inchannel, outchannel, kernel_size, stride, padding)
            return nn.Sequential(conv)
        else:
            conv = nn.Conv3d(inchannel, outchannel, kernel_size, stride, padding)
            batchnorm = nn.BatchNorm3d(outchannel)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, batchnorm, leakrelu)

    def forward(self, inp):
        x = self.conv_bolck1(inp)
        x = self.conv_bolok2(x)
        x = self.conv_bolok3(x)
        x = x.sigmoid()
        # x = x.view(x.size(0),-1)
        # x = x.mean(1,keepdim=True)
        return {'output': x}


class PatchGAN(nn.Module):
    def __init__(self, in_channels=1, ndf=32):
        super(PatchGAN, self).__init__()
        self.ndf = ndf
        channels = [ndf, ndf*2, ndf*4, ndf*8, ndf*16]
        kernel_size = 4
        stride = 2
        padding = 1
        # state size = 160
        self.conv_bolck1 = self.__conv_block(in_channels, channels[0], kernel_size, stride, padding, "first_layer")#128
        self.conv_bolok2 = self.__conv_block(channels[0], channels[1], kernel_size, stride, padding)#64
        self.conv_bolok3 = self.__conv_block(channels[1], 1, kernel_size, stride, padding, "last_layer")#32
    def __conv_block(self, inchannel, outchannel, kernel_size, stride, padding, layer=None):
        if layer == "first_layer":
            conv = nn.Conv3d(inchannel, outchannel, kernel_size, stride, padding)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, leakrelu)
        elif layer == "last_layer":
            conv = nn.Conv3d(inchannel, outchannel, kernel_size, stride, padding)
            return nn.Sequential(conv)
        else:
            conv = nn.Conv3d(inchannel, outchannel, kernel_size, stride, padding)
            batchnorm = nn.BatchNorm3d(inchannel)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(batchnorm, conv, leakrelu)

    def forward(self, inp):
        x = self.conv_bolck1(inp)
        x0 = self.conv_bolok2(x)
        x1 = self.conv_bolok3(x0)
        return {'output':x1.sigmoid(), 'feat':x0}


class Decoder(nn.Module):
    def __init__(self, in_channels=256, ndf=1):
        super(Decoder, self).__init__()
        self.conv1 = ResnetBlock(in_channels)
        self.conv2 = ResnetBlock(128)
        self.conv3 = ResnetBlock(64)
        self.ct1 = ConvTransposeBlock(in_channels,128)
        self.ct2 = ConvTransposeBlock(128,64)
        self.ct3 = ConvTransposeBlock(64,32)
        self.conv = nn.Conv3d(32, ndf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):
        x = self.conv1(feature)
        x = self.ct1(x)
        x = self.conv2(x)
        x = self.ct2(x)
        x = self.conv3(x)
        x = self.ct3(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """Resnet block"""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        # add convolutional layer followed by normalization and ReLU
        self.conv_block = nn.Sequential(nn.ReflectionPad3d(1),
                                         nn.Conv3d(dim, dim, kernel_size=3, padding=0),
                                         nn.InstanceNorm3d(dim),
                                         nn.LeakyReLU(0.2, True),
                                         nn.ReflectionPad3d(1),
                                         nn.Conv3d(dim, dim, kernel_size=3, padding=0),
                                         nn.InstanceNorm3d(dim))

    def forward(self, x):
        """Forward pass (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ConvTransposeBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True):
        super(ConvTransposeBlock, self).__init__()

        self.main = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out