# Reference
# https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/

import config
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MultiHeadSelfAttention(nn.Module):
	
	def __init__(self, in_dim, attn_dim, n_head, drop):

		'''
			in_dim  : input dimension
			attn_dim: attention dimension
			n_head : number of attention heads
			drop    : dropout, to prevent overfitting
		'''

		super(MultiHeadSelfAttention, self).__init__()

		head_dim           = attn_dim * n_head # Total output dimension including all the heads
		self.scale         = head_dim ** (-0.5)
		self.key_lyr       = self.get_qkv_layer(in_dim, head_dim, n_head)
		self.query_lyr     = self.get_qkv_layer(in_dim, head_dim, n_head)
		self.value_lyr     = self.get_qkv_layer(in_dim, head_dim, n_head)
		self.attn_score    = nn.Softmax(dim=-1)
		self.dropout_layer = nn.Dropout(drop)
		self.out_layer     = nn.Sequential(
											Rearrange('bsize n_head M head_dim -> bsize M (n_head head_dim)'),
											nn.Linear(in_features=head_dim, out_features=in_dim),
											nn.Dropout(drop)
										   )



	def get_qkv_layer(self, in_dim, head_dim, n_head):
		layer = nn.Sequential(
								nn.Linear(in_features=in_dim, out_features=head_dim, bias=False),
								Rearrange('bsize M (n_head head_dim) -> bsize n_head M head_dim', n_head=n_head)
							)
		return layer


	def forward(self, x):
		Q = self.key_lyr( x )
		K = self.query_lyr( x )
		V = self.value_lyr( x )

		attention = torch.einsum('b i j k, b i l k -> b i j l', Q, K) / self.scale
		scores    = self.attn_score(attention)
		scores    = self.dropout_layer(scores)
		weighted  = torch.einsum('b i j k, b i k l -> b i j l', scores, V)
		out       = self.out_layer( weighted )		

		return out

if __name__ == '__main__':

	inp  = torch.rand(size=(1, 196, 768))
	mhsa = MultiHeadSelfAttention(
									in_dim=config.in_dim,\
									attn_dim=config.attn_dim,\
									n_head=config.n_head,\
									drop=config.drop
								 )
	print('Input shape: {}'.format(inp.shape))
	print('Output shape: {}'.format(mhsa(inp).shape))