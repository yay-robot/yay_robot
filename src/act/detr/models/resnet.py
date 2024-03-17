import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from .efficientnet import FiLMBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        use_film=False,
        language_embed_size=768,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.use_film = use_film
        if self.use_film:
            self.film = FiLMBlock(
                language_embed_size=language_embed_size, num_channels=planes
            )

    def forward(self, x, language_embed=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_film and language_embed is not None:
            out = self.film(out, language_embed)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        use_film=False,
        language_embed_size=768,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.use_film = use_film
        if self.use_film:
            self.film = FiLMBlock(
                language_embed_size=language_embed_size,
                num_channels=planes * self.expansion,
            )

    def forward(self, x, language_embed=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_film and language_embed is not None:
            out = self.film(out, language_embed)

        out += identity
        out = self.relu(out)

        return out


class FilmedBasicBlock(BasicBlock):
    def forward(self, x, language_embed=None):
        if language_embed is not None:
            return filmed_forward(self, x, language_embed)
        return super().forward(x)


class FilmedBottleneck(Bottleneck):
    def forward(self, x, language_embed=None):
        if language_embed is not None:
            return filmed_forward(self, x, language_embed)
        return super().forward(x)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CustomResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, use_film=False
    ):
        if dilate:
            self.dilation *= stride
            stride = 1
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                use_film,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_film=use_film,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, language_embed=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass language_embed to layers if they use FiLM
        x = (
            self.layer1(x, language_embed)
            if isinstance(self.layer1[0], (BasicBlock, Bottleneck))
            and self.layer1[0].use_film
            else self.layer1(x)
        )
        x = (
            self.layer2(x, language_embed)
            if isinstance(self.layer2[0], (BasicBlock, Bottleneck))
            and self.layer2[0].use_film
            else self.layer2(x)
        )
        x = (
            self.layer3(x, language_embed)
            if isinstance(self.layer3[0], (BasicBlock, Bottleneck))
            and self.layer3[0].use_film
            else self.layer3(x)
        )
        x = (
            self.layer4(x, language_embed)
            if isinstance(self.layer4[0], (BasicBlock, Bottleneck))
            and self.layer4[0].use_film
            else self.layer4(x)
        )

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def filmed_basic_block():
    return FilmedBasicBlock


def filmed_bottleneck():
    return FilmedBottleneck


def filmed_forward(self, x, language_embed):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    # Apply FiLM before adding the residual connection
    film = FiLMBlock(
        language_embed_size=language_embed.size(1), num_channels=out.size(1)
    )
    out = film(out, language_embed)

    out += identity
    out = self.relu(out)

    return out


def _resnet(arch, block, layers, weights, use_film, **kwargs):
    model = CustomResNet(block, layers, **kwargs)
    if use_film:
        for n, m in model.named_modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                m.forward = lambda x, language_embed: filmed_forward(
                    m, x, language_embed
                )
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=True))
    return model


def film_resnet18(weights=None, use_film=True, **kwargs):
    return _resnet(
        "resnet18", filmed_basic_block(), [2, 2, 2, 2], weights, use_film, **kwargs
    )


def film_resnet34(weights=None, use_film=True, **kwargs):
    return _resnet(
        "resnet34", filmed_basic_block(), [3, 4, 6, 3], weights, use_film, **kwargs
    )


def film_resnet50(weights=None, use_film=True, **kwargs):
    return _resnet(
        "resnet50", filmed_bottleneck(), [3, 4, 6, 3], weights, use_film, **kwargs
    )
