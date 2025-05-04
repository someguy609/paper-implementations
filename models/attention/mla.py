'''
Multi-Head Latent Attention implementation in PyTorch. 
Based of the "TransMLA: Multi-Head Latent Attention Is All You Need" paper.
https://arxiv.org/pdf/2502.07864
'''

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from typing import Optional

@dataclass
class MLAConfig:
	model_dim: int
	num_heads: int

	q_lora_rank: int
	k_lora_rank: int
	v_lora_rank: int

	batch_size: int
	max_seq_len: int
	use_cache: Optional[bool] = False

	dropout: Optional[float] = 0

class MLA(nn.Module):

	def __init__(self, config: MLAConfig) -> None:
		super(MLA, self).__init__()
		assert config.model_dim % config.num_heads == 0

		self.model_dim = config.model_dim
		self.num_heads = config.num_heads
		self.q_lora_rank = config.q_lora_rank
		self.k_lora_rank = config.k_lora_rank
		self.v_lora_rank = config.v_lora_rank
		self.use_cache = config.use_cache
		self.dropout = config.dropout
		
		self.head_dim = self.model_dim // self.num_heads
		
		self.wq_a = nn.Linear(self.model_dim, self.q_lora_rank, bias=False)
		self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
		self.wk_a = nn.Linear(self.model_dim, self.k_lora_rank, bias=False)
		self.wk_b = nn.Linear(self.k_lora_rank, self.num_heads * self.head_dim, bias=False)
		self.wv_a = nn.Linear(self.model_dim, self.v_lora_rank, bias=False)
		self.wv_b = nn.Linear(self.v_lora_rank, self.num_heads * self.head_dim, bias=False)
		self.wo = nn.Linear(self.num_heads * self.head_dim, self.model_dim)
		
		if self.use_cache:
			self.register_buffer('k_cache', torch.zeros((config.batch_size, config.max_seq_len, self.k_lora_rank)))
			self.register_buffer('v_cache', torch.zeros((config.batch_size, config.max_seq_len, self.v_lora_rank)))
	
	def forward(
		self, x: torch.Tensor, 
		mask: Optional[torch.Tensor] = None,
		pos: Optional[int] = 0
	) -> torch.Tensor:
		b, t, d = x.shape

		c_q = self.wq_a(x)
		c_k = self.wk_a(x)
		c_v = self.wv_a(x)

		if not self.training and self.use_cache:
			self.k_cache[:b, pos:pos + t] = c_k
			self.v_cache[:b, pos:pos + t] = c_v
			c_k = self.k_cache[:b, :pos + t]
			c_v = self.v_cache[:b, :pos + t]

		q = self.wq_b(c_q).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
		k = self.wk_b(c_k).view(b, c_k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
		v = self.wv_b(c_v).view(b, c_v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

		# apply rope to q and k

		outputs = F.scaled_dot_product_attention(q, k, v, mask, self.dropout)
		outputs = outputs.transpose(1, 2)
		outputs = outputs.view(b, t, d)
		outputs = self.wo(outputs)
		
		return outputs