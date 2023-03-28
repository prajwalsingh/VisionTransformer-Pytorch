# Reference
# https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/

import config
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from mhsa import MultiHeadSelfAttention

class TransformerEncoder(nn.Module):
	def __init__(self, n_head, n_layer, embed_dim, head_dim, mlp_hid_dim, drop):
		'''
			n_head     : number of heads in MHSA layer
			n_layer    : number of MHSA layer in the transformer
			embed_dim  : dimension of input token
			head_dim   : dimensionality of each attention head
			mlp_hid_dim: number of neuron in mlp layer
			drop       : droprate
		'''
		super(TransformerEncoder, self).__init__()
		self.n_head      = n_head
		self.n_layer     = n_layer
		self.embed_dim   = embed_dim
		self.head_dim    = head_dim
		self.mlp_hid_dim = mlp_hid_dim
		self.drop        = drop

		self.self_attn_block, self.feed_forward_block = self.get_layers()

	def get_layers(self):
		self_attn_block    = nn.ModuleList()
		feed_forward_block = nn.ModuleList()

		for idx in range(self.n_layer):
			self_attn_block.append(nn.Sequential(
													nn.LayerNorm(self.embed_dim),
													MultiHeadSelfAttention(
																			self.embed_dim,
																			self.head_dim,
																			self.n_head,
																			self.drop
																		  )
												))

			feed_forward_block.append(nn.Sequential(
														nn.LayerNorm(self.embed_dim),
														nn.Linear(in_features=self.embed_dim, out_features=self.mlp_hid_dim),
														nn.GELU(),
														nn.Dropout(self.drop),
														nn.Linear(in_features=self.mlp_hid_dim, out_features=self.embed_dim),
														nn.Dropout(self.drop)
													))

		return self_attn_block, feed_forward_block

	def forward(self, x):
		for (self_attn_block, feed_forward_block) in zip(self.self_attn_block, self.feed_forward_block):
			x = x + self_attn_block( x )
			x = x + feed_forward_block( x )

		return x


if __name__ == '__main__':

	inp  = torch.rand(size=(1, 196, 768))
	encoder = TransformerEncoder(
								n_head=config.n_head,\
								n_layer=config.n_enc_layer,\
								embed_dim=config.embed_dim,\
								head_dim=config.head_dim,\
								mlp_hid_dim=config.mlp_hid_dim,\
								drop=config.drop
							 )
	print('Input shape: {}'.format(inp.shape))
	print('Output shape: {}'.format(encoder(inp).shape))