'''
VGG16 Implementation in PyTorch.
Based of the "Very Deep Convolutional Networks for Large-Scale Image Recognition" paper.
https://arxiv.org/abs/1409.1556
'''

import torch
from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from typing import Optional

class CReLU(nn.Sequential):

	def __init__(self, *args, **kwargs) -> None:
		super(CReLU, self).__init__(OrderedDict([
			('conv', nn.Conv2d(*args, **kwargs)),
			('act', nn.ReLU(inplace=True))
		]))

class LNReLU(nn.Sequential):

	def __init__(self, channels: int) -> None:
		super(LNReLU, self).__init__(OrderedDict([
			('norm', nn.LocalResponseNorm(channels)),
			('act', nn.ReLU(inplace=True))
		]))

@dataclass
class VGG16Config:
	layers: list[str]
	channels: Optional[int] = 3
	base: Optional[int] = 64
	num_classes: Optional[int] = 1000
	dropout: Optional[float] = 0

VGG16_A = VGG16Config(['3', '3', '33', '33', '33'])
VGG16_A_LRN = VGG16Config(['3n', '3', '33', '33', '33'])
VGG16_B = VGG16Config(['33', '33', '33', '33', '33'])
VGG16_C = VGG16Config(['33', '33', '331', '331', '331'])
VGG16_D = VGG16Config(['33', '33', '333', '333', '333'])
VGG16_E = VGG16Config(['33', '33', '3333', '3333', '3333'])

class VGG16(nn.Module):

	def __init__(self, config: VGG16Config) -> None:
		super(VGG16, self).__init__()
		self.features = nn.Sequential(
			self._make_layer(config.channels, config.base, config.layers[0]),
			self._make_layer(config.base, config.base * 2, config.layers[1]),
			self._make_layer(config.base * 2, config.base * 4, config.layers[2]),
			self._make_layer(config.base * 4, config.base * 8, config.layers[3]),
			self._make_layer(config.base * 8, config.base * 8, config.layers[4])
		)
		self.fc = nn.Sequential(
			nn.Linear(7 * 7 * config.base * 8, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(config.dropout, inplace=True),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(config.dropout, inplace=True),
			nn.Linear(4096, config.num_classes)
		)
	
	def _get_layer(
		self, 
		layer: str, 
		in_channels: int, 
		out_channels: int
	) -> nn.Module:
		if layer == '1':
			return CReLU(in_channels, out_channels, kernel_size=1)
		if layer == '3':
			return CReLU(in_channels, out_channels, kernel_size=3, padding=1)
		if layer == 'n':
			return LNReLU(in_channels)
		raise KeyError('Layer not found')
	
	def _make_layer(
		self, 
		in_channels: int, 
		out_channels: int, 
		config: str
	) -> nn.Module:
		layers = [ self._get_layer(config[0], in_channels, out_channels) ]
		layers += [ self._get_layer(c, out_channels, out_channels) for c in config[1:] ]
		layers += [ nn.MaxPool2d(kernel_size=2, stride=2) ]
		return nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x