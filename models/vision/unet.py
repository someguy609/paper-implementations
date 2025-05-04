'''
U-Net implementation in PyTorch.
Based of the "U-Net: Convolutional Networks for Biomedical Image Segmentation" paper.
https://arxiv.org/pdf/1505.04597
'''

import torch
from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from typing import Optional

@dataclass
class UNetConfig:
	channels: int
	base: Optional[int] = 64
	depth: Optional[int] = 4
	kernel_size: Optional[int] = 3
	padding: Optional[int] = 1

class CBR(nn.Sequential):

	def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
		super(CBR, self).__init__(OrderedDict([
			('conv', nn.Conv2d(in_channels, out_channels, *args, **kwargs)),
			('norm', nn.BatchNorm2d(out_channels)),
			('act', nn.ReLU(inplace=True))
		]))

class ConvBlock(nn.Sequential):

	def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
		super(ConvBlock, self).__init__(OrderedDict([
			('conv1', CBR(in_channels, out_channels, *args, **kwargs)),
			('conv2', CBR(out_channels, out_channels, *args, **kwargs))
		]))

class Downsample(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
		super(Downsample, self).__init__()
		self.features = ConvBlock(in_channels, out_channels, *args, **kwargs)
		self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
	
	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = self.features(x)
		skip = x
		x = self.downsample(x)
		return x, skip

class Upsample(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
		super(Upsample, self).__init__()
		self.features = ConvBlock(out_channels * 2, out_channels, *args, **kwargs)
		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
	
	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.upsample(x)
		x = torch.cat([skip, x], dim=1)
		x = self.features(x)
		return x

class UNet(nn.Module):

	def __init__(self, config: UNetConfig) -> None:
		super(UNet, self).__init__()
		self.channels = config.channels
		self.base = config.base
		self.depth = config.depth

		self.downsampling = nn.ModuleList([ Downsample(self.channels, self.base, kernel_size=config.kernel_size, padding=config.padding) ])
		for i in range(self.depth - 1):
			self.downsampling.append(Downsample(self.base * 2 ** i, self.base * 2 ** (i + 1), kernel_size=config.kernel_size, padding=config.padding))

		self.upsampling = nn.ModuleList([])
		for i in reversed(range(self.depth)):
			self.upsampling.append(Upsample(self.base * 2 ** (i + 1), self.base * 2 ** i, kernel_size=config.kernel_size, padding=config.padding))
		
		self.bottleneck = ConvBlock(self.base * 2 ** 3, self.base * 2 ** 4, kernel_size=config.kernel_size, padding=config.padding)
		self.head = nn.Conv2d(self.base, 1, kernel_size=1)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		skips = []
		for downsample in self.downsampling:
			x, skip = downsample(x)
			# center crop skip
			skips.append(skip)
		x = self.bottleneck(x)
		for upsample in self.upsampling:
			skip = skips.pop()
			x = upsample(x, skip)
		x = self.head(x)
		return x
