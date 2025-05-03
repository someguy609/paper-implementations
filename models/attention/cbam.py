'''
Convolutional Block Attention Module implementation in PyTorch.
Based on the "CBAM: Convolutional Block Attention Module" paper.
https://arxiv.org/abs/1807.06521
'''

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from typing import Optional

class CAM(nn.Module):

	def __init__(
		self, 
		channels: int, 
		reduction_ratio: int
	) -> None:
		super(CAM, self).__init__()
		self.mlp = nn.Sequential(
			nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		max_pool = F.adaptive_max_pool2d(x, 1)
		avg_pool = F.adaptive_avg_pool2d(x, 1)

		max_pool = self.mlp(max_pool)
		avg_pool = self.mlp(avg_pool)

		outputs = max_pool + avg_pool
		outputs = outputs.sigmoid()

		return outputs

class SAM(nn.Module):

	def __init__(
		self, 
		kernel_size: int
	) -> None:
		super(SAM, self).__init__()
		self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		max_pool = x.max(dim=1, keepdim=True)[0]
		avg_pool = x.mean(dim=1, keepdim=True)
		cat_pool = torch.cat([max_pool, avg_pool], dim=1)
		
		outputs = self.conv(cat_pool)
		outputs = outputs.sigmoid()

		return outputs

@dataclass
class CBAMConfig:
	channels: int
	cam_reduction_ratio: Optional[int] = 16
	sam_kernel_size: Optional[int] = 7

class CBAM(nn.Module):

	def __init__(self, config: CBAMConfig) -> None:
		super(CBAM, self).__init__()
		self.cam = CAM(config.channels, config.cam_reduction_ratio)
		self.sam = SAM(config.sam_kernel_size)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x *= self.cam(x)
		x *= self.sam(x)
		return x
