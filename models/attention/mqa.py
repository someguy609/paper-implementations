'''
Multi-Query Attention implementation in PyTorch.
Based of the "Fast Transformer Decoding: One Write-Head is All You Need" paper.
https://arxiv.org/abs/1911.02150
'''

import torch
from dataclasses import dataclass
from torch import nn
from typing import Optional

@dataclass
class MQAConfig:
	model_dim: int
	num_heads: int

class MQA(nn.Module):

	def __init__(self, config: MQAConfig) -> None:
		super(MQA, self).__init__()
		assert config.model_dim % config.num_heads == 0

		self.model_dim = config.model_dim
		self.num_heads = config.num_heads
		self.head_dim = config.model_dim // config.num_heads

		self.wq = nn.Linear(self.model_dim, self.num_heads * self.head_dim, bias=False)
		self.wk = nn.Linear(self.model_dim, self.head_dim, bias=False)
		self.wv = nn.Linear(self.model_dim, self.head_dim, bias=False)
		self.wo = nn.Linear(self.num_heads * self.head_dim, self.model_dim, bias=False)
	
	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		b, t, d = x.shape

		q = self.wq(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2) # (b, nh, t, hd)
		k = self.wk(x).view(b, t, 1, self.head_dim).transpose(1, 2) # (b, 1, t, hd)
		v = self.wv(x).view(b, t, 1, self.head_dim).transpose(1, 2) # (b, 1, t, hd)

		attn = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5 # (b, nh, t, t)

		if mask is not None:
			attn = attn.masked_fill(mask == 0, -1e-9) # (b, nh, t, t)
		
		attn = attn.softmax(dim=-1) # (b, nh, t, t)

		outputs = (attn @ v).transpose(1, 2) # (b, t, nh, hd)
		outputs = outputs.contiguous().view(b, t, d) # (b, t, d)
		outputs = self.wo(outputs) # (b, t, d)

		return outputs