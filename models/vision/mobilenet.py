'''
MobileNet implementation in PyTorch.
Based on:
- "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" for MobileNet (https://arxiv.org/abs/1704.04861)
- "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for MobileNetV2 (https://arxiv.org/abs/1801.04381)
- "Searching for MobileNetV3" for MobileNetV3 (https://arxiv.org/abs/1905.02244)
'''

import torch
from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from typing import Optional

class CBR(nn.Sequential):

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int, 
		*args, 
		**kwargs
	) -> None:
		super(CBR, self).__init__(OrderedDict([
			('conv', nn.Conv2d(in_channels, out_channels, *args, **kwargs)),
			('norm', nn.BatchNorm2d(out_channels)),
			('act', nn.ReLU(inplace=True))
		]))

class CBR6(nn.Sequential):

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int, 
		act: Optional[bool] = True,
		*args, 
		**kwargs
	) -> None:
		super(CBR6, self).__init__(OrderedDict([
			('conv', nn.Conv2d(in_channels, out_channels, *args, **kwargs)),
			('norm', nn.BatchNorm2d(out_channels)),
			('act', nn.ReLU6(inplace=True) if act else nn.Identity())
		]))

class DWConv(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
		super(DWConv, self).__init__()
		self.depthwise = CBR(in_channels, in_channels, groups=in_channels, *args, **kwargs)
		self.pointwise = CBR(in_channels, out_channels, kernel_size=1)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x

class Bottleneck(nn.Module):

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int, 
		stride: Optional[int] = 1, 
		expansion: Optional[int] = 1, 
		*args, **kwargs
	) -> None:
		super(Bottleneck, self).__init__()
		self.pointwise = CBR6(in_channels, expansion * in_channels, kernel_size=1)
		self.depthwise = CBR6(expansion * in_channels, expansion * in_channels, stride=stride, groups=in_channels * expansion, *args, **kwargs)
		self.linear = CBR6(expansion * in_channels, out_channels, kernel_size=1, act=False)
		self.shortcut = (
			CBR6(in_channels, out_channels, kernel_size=1, stride=stride, act=False)
			if in_channels != out_channels or stride != 1 else nn.Identity()
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.shortcut(x)
		x = self.pointwise(x)
		x = self.depthwise(x)
		x = self.linear(x)
		x += residual
		return x

@dataclass
class MobileNetConfig:
	layers: list[int]
	channels: Optional[int] = 3
	base: Optional[int] = 32
	num_classes: Optional[int] = 1000

MOBILENET = MobileNetConfig([1, 2, 2, 6, 2])
MOBILENETV2 = MobileNetConfig([1, 2, 3, 4, 3, 3, 1])

class MobileNet(nn.Module):

	def __init__(self, config: MobileNetConfig) -> None:
		super(MobileNet, self).__init__()
		self.in_channels = config.base
		self.features = nn.Sequential(
			CBR(config.channels, config.base, kernel_size=3, stride=2, padding=1), # 32
			self._make_layer(config.base * 2, config.layers[0]), # 64
			self._make_layer(config.base * 4, config.layers[1], stride=2), # 128
			self._make_layer(config.base * 8, config.layers[2], stride=2), # 256
			self._make_layer(config.base * 16, config.layers[3] - 1), # 512
			self._make_layer(config.base * 16, 1, stride=2), # 512
			self._make_layer(config.base * 32, config.layers[4], stride=2), # 1024
		)
		self.fc = nn.Linear(config.base * 32, config.num_classes)
	
	def _make_layer(self, out_channels: int, depth: int, stride: Optional[int] = 1) -> nn.Module:
		layers = [ DWConv(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1) ]
		self.in_channels = out_channels
		layers += [ DWConv(self.in_channels, out_channels, kernel_size=3, padding=1) for _ in range(depth - 1) ]
		return nn.Sequential(*layers)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = F.adaptive_avg_pool2d(x, 1)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class MobileNetV2(nn.Module):

	def __init__(self, config: MobileNetConfig) -> None:
		super(MobileNetV2, self).__init__()
		self.in_channels = config.base
		self.features = nn.Sequential(
			CBR6(config.channels, config.base, kernel_size=3, stride=2, padding=1), # 32
			self._make_layer(config.base // 2, config.layers[0], expansion=1), # 16
			self._make_layer(config.base * 3 // 4, config.layers[1], stride=2), # 24
			self._make_layer(config.base, config.layers[2], stride=2), # 32
			self._make_layer(config.base * 2, config.layers[3], stride=2), # 64
			self._make_layer(config.base * 3, config.layers[4]), # 96
			self._make_layer(config.base * 5, config.layers[5], stride=2), # 160
			self._make_layer(config.base * 10, config.layers[6]), # 320
			CBR6(config.base * 10, config.base * 40, kernel_size=1), # 1280
		)
		self.fc = nn.Linear(config.base * 40, config.num_classes)
	
	def _make_layer(
		self, 
		out_channels: int, 
		depth: int, 
		stride: Optional[int] = 1, 
		expansion: Optional[int] = 6
	) -> nn.Module:
		layers = [ Bottleneck(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, expansion=expansion) ]
		self.in_channels = out_channels
		layers += [ Bottleneck(self.in_channels, out_channels, kernel_size=3, padding=1, expansion=expansion) for _ in range(depth - 1) ]
		return nn.Sequential(*layers)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = F.adaptive_avg_pool2d(x, 1)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class MobileNetV3(nn.Module):

	def __init__(self) -> None:
		super(MobileNetV3, self).__init__()