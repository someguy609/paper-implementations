'''
MobileNet implementation in PyTorch.
Based on:
- https://arxiv.org/abs/1704.04861 for MobileNet
- https://arxiv.org/abs/1801.04381 for MobileNetV2
- https://arxiv.org/abs/1905.02244 for MobileNetV3
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

class DWConv(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
		super(DWConv, self).__init__()
		self.depthwise = CBR(in_channels, in_channels, groups=in_channels, *args, **kwargs)
		self.pointwise = CBR(in_channels, out_channels, kernel_size=1)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x

@dataclass
class MobileNetConfig:
	layers: list[int]
	channels: Optional[int] = 3
	base: Optional[int] = 32
	num_classes: Optional[int] = 1000

MOBILENET = MobileNetConfig()

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

	def __init__(self) -> None:
		super(MobileNetV2, self).__init__()

class MobileNetV3(nn.Module):

	def __init__(self) -> None:
		super(MobileNetV3, self).__init__()