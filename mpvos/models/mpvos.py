import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MPVOS(nn.Module):
    def __init__(self, backbone, pretrained):
        super(MPVOS, self).__init__()

        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            channels = [256, 512, 1024, 2048]
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )

        self.layers = nn.ModuleList()
        self.layers.add_module(name="encode1", module=self.backbone.layer1)
        self.layers.add_module(name="encode2", module=self.backbone.layer2)
        self.layers.add_module(name="encode3", module=self.backbone.layer3)
        self.layers.add_module(name="encode4", module=self.backbone.layer4)

        self.attentions = nn.Sequential()
        # self.attentions.add_module(name="attention1", module=ParallelSecondOrderNet(in_channel=channels[0], out_channel=channels[0]))
        self.attentions.add_module(name="attention2", module=ParallelSecondOrderNet(in_channel=channels[1], out_channel=channels[1]))
        self.attentions.add_module(name="attention3", module=ParallelSecondOrderNet(in_channel=channels[2], out_channel=channels[2]))
        # self.attentions.add_module(name="attention4", module=ParallelSecondOrderNet(in_channel=channels[3], out_channel=channels[3]))

        self.convs = nn.Sequential()
        self.convs.add_module(name="down1", module=nn.Conv2d(channels[0]*2, channels[0], 3, 1, 1))
        self.convs.add_module(name="down4", module=nn.Conv2d(channels[3]*2, channels[3], 3, 1, 1))
        # decode
        emb_chan = 768
        self.skips = nn.ModuleList([nn.Conv2d(channel, emb_chan, 1, 1, 0) for channel in reversed(channels)])
        self.conv_seg = nn.Conv2d(emb_chan, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, image1, image2):

        orin_size = image1.shape[-2:]

        fuse_features = {}
        image1, image2 = self.stem(image1), self.stem(image2)
        c1, c2 = 0, 0
        for i, layer in enumerate(self.layers):
            image1 = layer(image1)
            image2 = layer(image2)
            if i in [0, 3]:
                fuse_feature = self.convs[c1](torch.cat([image1, image2], dim=1))
                c1 += 1
            else:
                fuse_feature = self.attentions[c2](image1, image2)
                c2 += 1
            fuse_features[f"fuse_feature{len(self.layers) - (i + 1)}"] = fuse_feature

        # decode
        for i in range(len(self.skips)):
            if i == 0:
                x = self.skips[i](fuse_features[f"fuse_feature{i}"])
            else:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + self.skips[i](fuse_features[f"fuse_feature{i}"])

        x = self.conv_seg(x)
        x = F.interpolate(x, size=orin_size, mode="bilinear", align_corners=False)

        return x


class TripleAttention(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=32):
        super(TripleAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_c = nn.AdaptiveAvgPool2d(1)

        mid_channel = max(8, in_channel // reduction)

        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channel)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_c = nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        identity = x
        # b c h 1
        x_h = self.pool_h(x)
        # b c 1 w
        x_w = self.pool_w(x)
        # b c 1 1
        x_c = self.pool_c(x)

        a_h = self.conv_h(self.act(self.bn1(self.conv1(x_h)))).sigmoid()
        a_w = self.conv_w(self.act(self.bn2(self.conv2(x_w)))).sigmoid()
        a_c = self.conv_c(self.act(self.bn3(self.conv3(x_c)))).sigmoid()

        out = identity * a_w * a_h * a_c

        return out


class ParallelSecondOrderNet(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=32):
        super(ParallelSecondOrderNet, self).__init__()

        self.first_branch = nn.Sequential(
            ParallelSecondOrderModule(in_channel, out_channel, reduction),
            ResBlock(out_channel)
        )
        self.second_branch = nn.Sequential(
            ParallelSecondOrderModule(in_channel, out_channel, reduction),
            ResBlock(out_channel)
        )
        self.triple_attention = TripleAttention(in_channel=in_channel, out_channel=out_channel, reduction=reduction)
        self.out = nn.Conv2d(out_channel * 2 + 1, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, image1, image2):
        # b c h w
        identity = image2
        # b c h w -> b h c w
        image1 = self.first_branch(image1).permute(0, 2, 1, 3)
        # b c h w -> b h w c
        image2 = self.second_branch(image2).permute(0, 2, 3, 1)
        # b h w w
        mid = torch.matmul(image2, image1)
        # b h 1 w
        mask = torch.softmax(mid.permute(0, 1, 3, 2), dim=-1).sum(dim=-2, keepdim=True)
        mask = mask.permute(0, 2, 1, 3)
        mask[mask>=0.1] = 1
        mask[mask<0.1] = 0
        a2 = torch.softmax(mid, dim=-1)
        image2 = torch.matmul(a2, identity.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        image3 = self.triple_attention(identity) * identity
        out = self.out(torch.cat([mask, image2, image3], dim=1))

        return out


class ParallelSecondOrderModule(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=32):
        super(ParallelSecondOrderModule, self).__init__()

        mid_channel = max(8, in_channel // reduction)

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )

        self.conv = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        identity = x
        x1 = x
        x2 = x
        # b c h w
        x1_mean = torch.mean(x1, dim=1, keepdim=True)
        x1_max, _ = torch.max(x1, dim=1, keepdim=True)
        # b 2 h w
        x1 = torch.cat([x1_mean, x1_max], dim=1)
        x1 = self.layer1(x1).sigmoid()
        x1 = x1 * identity
        x2 = self.layer2(x2)
        x2 = x2 * identity
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

    def forward(self, x):
        return self.body(x) + x


if __name__ == '__main__':

    x = torch.randn(2, 3, 512, 512)
    net = MPVOS(backbone="resnet50", pretrained=False)
    net(x, x)