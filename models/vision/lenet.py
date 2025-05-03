'''
LeNet5 implementation in PyTorch.
Based of the "Paper: Gradient-Based Learning Applied to Document Recognition" paper.
https://ieeexplore.ieee.org/document/726791
'''

import torch
from dataclasses import dataclass
from torch import nn
from typing import Optional

@dataclass
class LeNet5Config:
	channels: Optional[int] = 1
	num_classes: Optional[int] = 10

class LeNet5(nn.Module):

	def __init__(self, config: LeNet5Config) -> None:
		super(LeNet5, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(config.channels, 6, kernel_size=5), 
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2, stride=2),
			nn.Conv2d(6, 16, kernel_size=5), 
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 120, kernel_size=5), 
			nn.Tanh(),
		)
		self.fc = nn.Sequential(
			nn.Linear(120, 84),
			nn.Tanh(),
			nn.Linear(84, config.num_classes)
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
