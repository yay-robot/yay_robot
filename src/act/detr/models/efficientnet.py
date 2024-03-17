"""
This code is mostly copied from the source below, with modifications to allow for FiLM language conditioning.
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
"""
import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import (
    _make_divisible,
    _ovewrite_named_param,
    handle_legacy_interface,
)


class FiLMBlock(nn.Module):
    def __init__(self, language_embed_size, num_channels):
        """FiLM conditioning block for a convolutional neural network.

        Args:
            language_embed_size: Dimensionality of the language embedding used to condition the network.
            num_channels: Number of convolutional channels in the convolution block being FiLMed.
        """
        super(FiLMBlock, self).__init__()
        self.scale = nn.Linear(language_embed_size, num_channels)
        self.shift = nn.Linear(language_embed_size, num_channels)

    def forward(self, x, language_embed):
        # x shape: (batch_size, C, H, W)
        # gamma shape: (batch_size, C)
        # beta shape: (batch_size, C)
        gamma = self.scale(language_embed)
        beta = self.shift(language_embed)
        # Instead of x = x * gamma + beta, we do x = x * (1 + gamma) + beta because if gamma is initially zero-centered,
        # then the former would zero-out the CNN feature map activations and thus gradients.
        # On the other hand, the latter formulation makes the transformation close to identity.
        # Therefore, the original paper actually uses the latter.
        # See Section 7.2 in the FiLM paper for further explanations: https://arxiv.org/pdf/1709.07871.pdf
        x = x * (1 + gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)) + beta.view(
            beta.shape[0], beta.shape[1], 1, 1
        )
        return x


__all__ = [
    "EfficientNet",
    "EfficientNet_B0_Weights",
    "EfficientNet_B1_Weights",
    "EfficientNet_B2_Weights",
    "EfficientNet_B3_Weights",
    "EfficientNet_B4_Weights",
    "EfficientNet_B5_Weights",
    "EfficientNet_B6_Weights",
    "EfficientNet_B7_Weights",
    "EfficientNet_V2_S_Weights",
    "EfficientNet_V2_M_Weights",
    "EfficientNet_V2_L_Weights",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(
        channels: int, width_mult: float, min_value: Optional[int] = None
    ) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        use_film: bool,
        language_embed_size: int,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

        self.use_film = use_film
        if self.use_film:
            self.film = FiLMBlock(
                language_embed_size=language_embed_size, num_channels=self.out_channels
            )

    def forward(self, input: Tensor, language_embed: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        if self.use_film:
            result = self.film(result, language_embed)
        return result


class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        use_film: bool,
        language_embed_size: int,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

        self.use_film = use_film
        if self.use_film:
            self.film = FiLMBlock(
                language_embed_size=language_embed_size, num_channels=self.out_channels
            )

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        if self.use_film:
            result = self.film(result, language_embed)
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        use_film: bool = True,
        language_embed_size: int = 768,  # 512 / 768
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
            use_film (bool): Whether to use FiLM language conditioning
            language_embed_size (int): The number of dimensions in the language embeddings to condition on
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        # layers.append(
        #     Conv2dNormActivation(
        #         3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
        #     )
        # )
        self.first_layer = Conv2dNormActivation(
            3,
            firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )

        # building inverted residual blocks
        self.inverted_residual_block_stages = nn.ModuleList()
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(
                    block_cnf.block(
                        block_cnf, sd_prob, norm_layer, use_film, language_embed_size
                    )
                )
                stage_block_id += 1

            # layers.append(nn.Sequential(*stage))
            self.inverted_residual_block_stages.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        # layers.append(
        #     Conv2dNormActivation(
        #         lastconv_input_channels,
        #         lastconv_output_channels,
        #         kernel_size=1,
        #         norm_layer=norm_layer,
        #         activation_layer=nn.SiLU,
        #     )
        # )
        self.last_layer = Conv2dNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )

        # self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

        # Initialize FiLM weights to zeros, as done in the RT-1 paper, so that adding FiLM doesn't disrupt intermediate activations
        # and negate the advantages of using pretrained weights.
        # Source: https://arxiv.org/pdf/2212.06817.pdf
        for n, m in self.named_modules():
            if isinstance(m, FiLMBlock):
                nn.init.zeros_(m.scale.weight)
                nn.init.zeros_(m.scale.bias)

    def _forward_impl(self, x: Tensor, language_embed: Tensor) -> Tensor:
        # x = self.features(x)
        x = self.first_layer(x)
        for stage in self.inverted_residual_block_stages:
            for filmed_mbconv_block in stage:
                x = filmed_mbconv_block(x, language_embed)
        x = self.last_layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor, language_embed: Tensor) -> Tensor:
        return self._forward_impl(x, language_embed)


def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    use_film: bool,
    **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        use_film=use_film,
        **kwargs,
    )

    if weights is not None:
        weights_dict = weights.get_state_dict(progress=True)
        # Since we changed the names of the parameter attributes in EfficientNet to integrate FiLM, we need to write our own custom pretrained weights-loading logic.
        # Standard (PyTorch-provided) pretrained parameters look like the first column below, but we want to map them to the names in the second column:
        #
        # ==========================================================================================
        # Pretrained parameters           ->            Current parameters
        # ==========================================================================================
        # features.0.0.weight                           first_layer.0.weight
        # features.0.1.weight                           first_layer.1.weight
        # features.0.1.bias                             first_layer.1.bias
        # features.0.1.running_mean                     first_layer.1.running_mean
        # features.0.1.running_var                      first_layer.1.running_var
        # features.0.1.num_batches_tracked              first_layer.1.num_batches_tracked
        # features.1.0.block.0.0.weight                 inverted_residual_block_stages.0.0.block.0.0.weight
        # features.1.0.block.0.1.weight                 inverted_residual_block_stages.0.0.block.0.1.weight
        # features.1.0.block.0.1.bias                   inverted_residual_block_stages.0.0.block.0.1.bias
        # ...
        # features.7.1.block.3.1.weight                 inverted_residual_block_stages.6.1.block.3.1.weight
        # features.7.1.block.3.1.bias                   inverted_residual_block_stages.6.1.block.3.1.bias
        # features.7.1.block.3.1.running_mean           inverted_residual_block_stages.6.1.block.3.1.running_mean
        # features.7.1.block.3.1.running_var            inverted_residual_block_stages.6.1.block.3.1.running_var
        # features.7.1.block.3.1.num_batches_tracked    inverted_residual_block_stages.6.1.block.3.1.num_batches_tracked
        # features.8.0.weight                           last_layer.0.weight
        # features.8.1.weight                           last_layer.1.weight
        # features.8.1.bias                             last_layer.1.bias
        # features.8.1.running_mean                     last_layer.1.running_mean
        # features.8.1.running_var                      last_layer.1.running_var
        # features.8.1.num_batches_tracked              last_layer.1.num_batches_tracked
        # classifier.1.weight                           (same as original)
        # classifier.1.bias                             (same as original)
        # ==========================================================================================
        #
        # The endpoint of the features.X.Y.____ is unknown, i.e., it doesn't always end as features.8.1.____ and instead depends on which EfficientNet variant we use.
        # We can get the endpoint by sorting the parameter names and checking the third-to-last parameter (fortunately, the parameter dict is already sorted).
        # Afterwards, a simple way to implement the mapping, as done below, is to hard-code the mapping of the features.0.Y.____ to first_layer.Y.____.
        # Then, we map features.X.Y.____ to inverted_residual_block_stages.{X-1}.Y.____ for X < endpoint.
        # Then, we map features.X.Y.____ to last_layer.Y.____ for X == endpoint.
        # Lastly, we leave the classifier parameters as is.
        weights_dict_keys = list(weights_dict.keys())
        endpoint = int(weights_dict_keys[-3].split(".")[1])  # e.g. 8
        old_to_new_key_mapping = OrderedDict()
        film_weights_in_weights_dict = (
            "inverted_residual_block_stages.0.0.film.scale.weight" in weights_dict_keys
        )
        if film_weights_in_weights_dict:
            # If FiLM weights are already included the weights, this means that the pretrained weights came from us (i.e., they're not provided by PyTorch).
            # In this case, we can load the weights as is.
            pass
        else:
            for k, v in weights_dict.items():
                if "features.0" in k:
                    new_key = (
                        "first_layer" + k.split("features.0")[1]
                    )  # hard-coded mapping for the first layer
                elif "features." in k:
                    features_id = int(k.split(".")[1])
                    if features_id < endpoint:
                        new_key = (
                            f"inverted_residual_block_stages.{features_id - 1}"
                            + k.split(f"features.{features_id}")[1]
                        )  # mapping for middle layers
                    elif features_id == endpoint:
                        new_key = (
                            "last_layer" + k.split(f"features.{endpoint}")[1]
                        )  # hard-coded mapping for the last layer
                    else:
                        raise ValueError(f"This branch should never be reached.")
                elif "classifier" in k:
                    new_key = k
                else:
                    raise ValueError(f"This branch should never be reached.")
                old_to_new_key_mapping[k] = new_key
            for old_key, new_key in old_to_new_key_mapping.items():
                weights_dict[new_key] = weights_dict.pop(old_key)
            # load_state_dict() will complain if the FiLM weights are not included in the weights dict.
            # Therefore, we can just copy over the current (freshly initialized) FiLM parameters into the weights dict before loading it.
            for name, param in model.named_parameters():
                if "film" in name:
                    weights_dict[name] = param
        model.load_state_dict(weights_dict)
    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}


_COMMON_META_V1 = {
    **_COMMON_META,
    "min_size": (1, 1),
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
}


_COMMON_META_V2 = {
    **_COMMON_META,
    "min_size": (33, 33),
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2",
}


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B0_Weights.IMAGENET1K_V1))
def film_efficientnet_b0(
    *,
    weights: Optional[EfficientNet_B0_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B0_Weights
        :members:
    """
    weights = EfficientNet_B0_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        use_film,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B1_Weights.IMAGENET1K_V1))
def film_efficientnet_b1(
    *,
    weights: Optional[EfficientNet_B1_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B1_Weights
        :members:
    """
    weights = EfficientNet_B1_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b1", width_mult=1.0, depth_mult=1.1
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        use_film,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B2_Weights.IMAGENET1K_V1))
def film_efficientnet_b2(
    *,
    weights: Optional[EfficientNet_B2_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B2_Weights
        :members:
    """
    weights = EfficientNet_B2_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b2", width_mult=1.1, depth_mult=1.2
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        use_film,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B3_Weights.IMAGENET1K_V1))
def film_efficientnet_b3(
    *,
    weights: Optional[EfficientNet_B3_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B3_Weights
        :members:
    """
    weights = EfficientNet_B3_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        use_film,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B4_Weights.IMAGENET1K_V1))
def film_efficientnet_b4(
    *,
    weights: Optional[EfficientNet_B4_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B4_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B4_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B4_Weights
        :members:
    """
    weights = EfficientNet_B4_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b4", width_mult=1.4, depth_mult=1.8
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        weights,
        progress,
        use_film,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B5_Weights.IMAGENET1K_V1))
def film_efficientnet_b5(
    *,
    weights: Optional[EfficientNet_B5_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B5_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B5_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B5_Weights
        :members:
    """
    weights = EfficientNet_B5_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b5", width_mult=1.6, depth_mult=2.2
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        weights,
        progress,
        use_film=use_film,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B6_Weights.IMAGENET1K_V1))
def film_efficientnet_b6(
    *,
    weights: Optional[EfficientNet_B6_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B6_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B6_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B6_Weights
        :members:
    """
    weights = EfficientNet_B6_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b6", width_mult=1.8, depth_mult=2.6
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        weights,
        progress,
        use_film=use_film,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B7_Weights.IMAGENET1K_V1))
def film_efficientnet_b7(
    *,
    weights: Optional[EfficientNet_B7_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B7_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B7_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B7_Weights
        :members:
    """
    weights = EfficientNet_B7_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b7", width_mult=2.0, depth_mult=3.1
    )
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        weights,
        progress,
        use_film=use_film,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@register_model()
@handle_legacy_interface(
    weights=("pretrained", EfficientNet_V2_S_Weights.IMAGENET1K_V1)
)
def film_efficientnet_v2_s(
    *,
    weights: Optional[EfficientNet_V2_S_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:
    """
    weights = EfficientNet_V2_S_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        use_film=use_film,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


@register_model()
@handle_legacy_interface(
    weights=("pretrained", EfficientNet_V2_M_Weights.IMAGENET1K_V1)
)
def film_efficientnet_v2_m(
    *,
    weights: Optional[EfficientNet_V2_M_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_M_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_M_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_M_Weights
        :members:
    """
    weights = EfficientNet_V2_M_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        use_film=use_film,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


@register_model()
@handle_legacy_interface(
    weights=("pretrained", EfficientNet_V2_L_Weights.IMAGENET1K_V1)
)
def film_efficientnet_v2_l(
    *,
    weights: Optional[EfficientNet_V2_L_Weights] = None,
    progress: bool = True,
    use_film: bool = True,
    **kwargs: Any,
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_L_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_L_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        use_film (bool, optional): If True, uses FiLM language conditioning.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_L_Weights
        :members:
    """
    weights = EfficientNet_V2_L_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        weights,
        progress,
        use_film=use_film,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )
