# Reference
# https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/

import config
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from encoder import TransformerEncoder
from torchsummary import summary


class VisionTransformer(nn.Module):
	def __init__(self, cfg):
		super(VisionTransformer, self).__init__()
		self.in_size            = cfg['input_size']
		self.patch_size         = cfg['patch_size']
		self.embed_dim          = cfg['embed_dim']
		self.n_enc_layer        = cfg['n_enc_layer']
		self.n_head             = cfg['n_head']
		self.head_dim           = cfg['head_dim']
		self.mlp_hid_dim        = cfg['mlp_hid_dim']
		self.drop               = cfg['drop']
		self.n_classes          = cfg['n_classes']

		self.num_patches = (self.in_size[1]//self.patch_size) *\
						 (self.in_size[2]//self.patch_size) +\
						 1 # num_of_patches + classification token

		self.patch_embedding = nn.Sequential(
												Rearrange('b c (h px) (w py) -> b (h w) (px py c)', px=self.patch_size, py=self.patch_size),
												nn.Linear(in_features=self.in_size[0] * (self.patch_size**2), out_features=self.embed_dim)
											) # Embedding without activation

		self.dropout_layer   = nn.Dropout(p=self.drop)
		self.cls_token       = nn.Parameter(torch.randn(1, 1, self.embed_dim))
		self.positional_embd = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim)) # Lernable positional encoding
		self.encoder         = TransformerEncoder(
													n_head=self.n_head,\
													n_layer=self.n_enc_layer,\
													embed_dim=self.embed_dim,\
													head_dim=self.head_dim,\
													mlp_hid_dim=self.mlp_hid_dim,\
													drop=self.drop
												 )
		self.prediction_head = nn.Sequential(
												nn.LayerNorm(self.embed_dim),
												nn.Linear(in_features=self.embed_dim, out_features=self.n_classes)
											)

	def forward(self, x):
		embed = self.patch_embedding( x )
		x     = torch.cat([torch.tile(self.cls_token, (x.shape[0], 1, 1)), embed], dim=1)
		x     = x + self.positional_embd
		x     = self.dropout_layer( x )
		x     = self.encoder( x )
		x     = x[:, 0, :] # Using only first token for classification
		pred  = self.prediction_head( x )
		return pred

if __name__ == '__main__':
	
	inp   = torch.rand(size=(2, 3, 224, 224))
	vit   = VisionTransformer(config.base).to(config.device)
	# y_cap = vit(inp)
	# print(y_cap.shape)
	summary(vit, input_size = (3, 224, 224), batch_size = -1)
