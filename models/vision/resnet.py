'''
ResNet implementation in PyTorch. 
Based of the "Deep Residual Learning for Image Recognition" paper.
https://arxiv.org/abs/1512.03385
'''

import torch
from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from typing import Optional, Union

class CBR(nn.Sequential):

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int,
		act: Optional[bool] = True,
		*args,
		**kwargs
	) -> None:
		super(CBR, self).__init__(OrderedDict([
			('conv', nn.Conv2d(in_channels, out_channels, *args, **kwargs)),
			('norm', nn.BatchNorm2d(out_channels)),
			('act', nn.ReLU(inplace=True) if act else nn.Identity())
		]))

class BasicBlock(nn.Module):

	expansion = 1

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int, 
		stride: Optional[int] = 1
	) -> None:
		super(BasicBlock, self).__init__()
		self.conv1 = CBR(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.conv2 = CBR(out_channels, out_channels, kernel_size=3, padding=1, act=False)
		self.shortcut = (
			CBR(in_channels, out_channels, kernel_size=1, stride=stride)
			if in_channels != out_channels or stride != 1 else nn.Identity()
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.shortcut(x)
		x = self.conv1(x)
		x = F.relu(self.conv2(x) + residual)
		return x

class Bottleneck(nn.Module):

	expansion: int = 4

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int, 
		stride: Optional[int] = 1
	) -> None:
		super(Bottleneck, self).__init__()
		self.conv1 = CBR(in_channels, in_channels, kernel_size=1)
		self.conv2 = CBR(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)
		self.conv3 = CBR(in_channels, out_channels * self.expansion, kernel_size=1, act=False)
		self.shortcut = (
			CBR(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, act=False)
			if in_channels != out_channels * self.expansion or stride != 1 else nn.Identity()
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.shortcut(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = F.relu(self.conv3(x) + residual)
		return x

@dataclass
class ResNetConfig:
	block: type[Union[BasicBlock, Bottleneck]]
	layers: list[int]
	channels: Optional[int] = 3
	base: Optional[int] = 64
	num_classes: Optional[int] = 1000

RESNET18 = ResNetConfig(BasicBlock, [2, 2, 2, 2])
RESNET34 = ResNetConfig(BasicBlock, [3, 4, 6, 3])
RESNET50 = ResNetConfig(Bottleneck, [3, 4, 6, 3])
RESNET101 = ResNetConfig(Bottleneck, [3, 4, 23, 3])
RESNET152 = ResNetConfig(Bottleneck, [3, 8, 36, 3])

class ResNet(nn.Module):

	def __init__(self, config: ResNetConfig) -> None:
		super(ResNet, self).__init__()
		self.in_channels = config.base
		self.features = nn.Sequential(
			nn.Conv2d(config.channels, config.base, kernel_size=7, stride=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			self._make_layer(config.block, config.base, config.layers[0], stride=1),
			self._make_layer(config.block, config.base * 2, config.layers[1]),
			self._make_layer(config.block, config.base * 4, config.layers[2]),
			self._make_layer(config.block, config.base * 8, config.layers[3])
		)
		self.fc = nn.Linear(self.in_channels, config.num_classes)
	
	def _make_layer(
		self, 
		block: type[Union[BasicBlock, Bottleneck]],
		channels: int,
		depth: int,
		stride: Optional[int] = 2,
	) -> nn.Module:
		layers = [ block(self.in_channels, channels, stride=stride) ]
		self.in_channels = channels * block.expansion
		layers += [ block(self.in_channels, channels) for _ in range(depth - 1) ]
		return nn.Sequential(*layers)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = F.adaptive_avg_pool2d(x, 1)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x