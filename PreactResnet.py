import jax
import flax
from flax import linen as nn

from functools import partial

from typing import Any, Sequence, Optional
import chex


class PreActBlock(nn.Module):
    """Pre-activation version of the Basic Block"""
    in_planes: int
    planes: int
    stride: int = 1
    expansion: int = 1
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> Any:
        train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.BatchNorm()(x=x, use_running_average=not train)
        out = nn.relu(x=out)

        if self.stride != 1 or self.in_planes != (self.expansion * self.planes):
            shortcut = nn.Conv(features=self.expansion * self.planes, kernel_size=(1, 1), strides=self.stride, use_bias=False)(inputs=x)
        else:
            shortcut = x

        out = nn.Conv(features=self.planes, kernel_size=(3, 3), strides=self.stride, padding=1, use_bias=False)(inputs=out)
        out = nn.BatchNorm()(x=out, use_running_average=not train)
        out = nn.relu(x=out)

        out = nn.Conv(features=self.planes, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=out)

        out = out + shortcut

        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    in_planes: int  # number of input channels
    planes: int  # number of output channels
    stride: int
    expansion: int = 4

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.BatchNorm()(x=x, use_running_average=not train)
        out = nn.relu(x=out)

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            shortcut = nn.Conv(features=self.expansion * self.planes, kernel_size=(1, 1), strides=self.stride, use_bias=False)(inputs=x)
        else:
            shortcut = x

        out = nn.Conv(features=self.planes, kernel_size=(1, 1), strides=1, padding=0, use_bias=False)(inputs=out)

        out = nn.BatchNorm()(x=out, use_running_average=not train)
        out = nn.relu(x=out)
        out = nn.Conv(features=self.planes, kernel_size=(3, 3), strides=self.stride, padding=1, use_bias=False)(inputs=out)

        out = nn.BatchNorm()(x=out, use_running_average=not train)
        out = nn.relu(x=out)
        out = nn.Conv(features=self.expansion * self.planes, kernel_size=(1, 1), use_bias=False)(inputs=out)

        out = out + shortcut

        return out


class PreActResNetFeature(nn.Module):
    """Extract features from a Resnet
    """
    block: PreActBlock | PreActBottleneck
    num_blocks: Sequence[int]
    in_planes: int = 64

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        out = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=x)

        in_planes, layer1 = make_layer(block=self.block, in_planes=self.in_planes, planes=64, num_blocks=self.num_blocks[0], stride=1, train=train)
        in_planes, layer2 = make_layer(block=self.block, in_planes=in_planes, planes=128, num_blocks=self.num_blocks[1], stride=2, train=train)
        in_planes, layer3 = make_layer(block=self.block, in_planes=in_planes, planes=256, num_blocks=self.num_blocks[2], stride=2, train=train)
        _, layer4 = make_layer(block=self.block, in_planes=in_planes, planes=512, num_blocks=self.num_blocks[3], stride=2, train=train)

        out = layer1(out)
        out = layer2(out)
        out = layer3(out)
        out = layer4(out)

        out = nn.avg_pool(inputs=out, window_shape=(4, 4), strides=(4, 4))
        out = out.reshape((out.shape[0], -1))

        return out


class Classifier(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        out = nn.Dense(features=self.num_classes)(inputs=x)
        return out


class PreActResNet(nn.Module):
    block: PreActBlock | PreActBottleneck
    num_blocks: Sequence[int]
    num_classes: int = 10
    in_planes: int = 64

    # @nn.compact
    # def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
    #     out = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=x)

    #     in_planes, layer1 = make_layer(block=self.block, in_planes=self.in_planes, planes=64, num_blocks=self.num_blocks[0], stride=1, train=train)
    #     in_planes, layer2 = make_layer(block=self.block, in_planes=in_planes, planes=128, num_blocks=self.num_blocks[1], stride=2, train=train)
    #     in_planes, layer3 = make_layer(block=self.block, in_planes=in_planes, planes=256, num_blocks=self.num_blocks[2], stride=2, train=train)
    #     _, layer4 = make_layer(block=self.block, in_planes=in_planes, planes=512, num_blocks=self.num_blocks[3], stride=2, train=train)

    #     out = layer1(out)
    #     out = layer2(out)
    #     out = layer3(out)
    #     out = layer4(out)

    #     out = nn.avg_pool(inputs=out, window_shape=(4, 4), strides=(4, 4))
    #     out = out.reshape((out.shape[0], -1))
    #     out = nn.Dense(features=self.num_classes)(inputs=out)

    #     return out

    def setup(self) -> None:
        self.features = PreActResNetFeature(block=self.block, num_blocks=self.num_blocks, in_planes=self.in_planes)
        self.classifier = Classifier(num_classes=self.num_classes)
        return None

    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        out = self.features(x=x, train=train)
        out = self.classifier(x=out)
        return out


def ResNet18(num_classes: int = 10) -> PreActResNet:
    return PreActResNet(block=PreActBlock, num_blocks=(2, 2, 2, 2), num_classes=num_classes)


def make_layer(block: PreActBlock | PreActBottleneck, in_planes: int, planes: int, num_blocks: Sequence[int], stride: int, train: bool) -> tuple[int, nn.Module]:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(
            partial(block, in_planes=in_planes, planes=planes, stride=stride, train=train)()
        )
        in_planes = planes * block.expansion

    return in_planes, nn.Sequential(layers=layers)
