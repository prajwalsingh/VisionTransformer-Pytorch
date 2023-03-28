# Reference:
# Base MNIST Parameter: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/

base_mnist={
		'input_size': [1, 28, 28],
		'patch_size': 7,
		'embed_dim': 8,
		'n_enc_layer': 2,
		'n_head': 2,
		'head_dim': 8,
		'mlp_hid_dim': 64,
		'drop': 0.1,
		'n_classes': 10
	}

base={
		'input_size': [3, 224, 224],
		'patch_size': 16,
		'embed_dim': 768,
		'n_enc_layer': 12,
		'n_head': 12,
		'head_dim': 64,
		'mlp_hid_dim': 3072,
		'drop': 0.1,
		'n_classes': 1000
	}

large={
		'input_size': [3, 224, 224],
		'patch_size': 16,
		'embed_dim': 1024,
		'n_enc_layer': 24,
		'n_head': 16,
		'head_dim': 64,
		'mlp_hid_dim': 4096,
		'drop': 0.1,
		'n_classes': 1000
	}

huge={
		'input_size': [3, 224, 224],
		'patch_size': 14,
		'embed_dim': 1280,
		'n_enc_layer': 32,
		'n_head': 16,
		'head_dim': 80,
		'mlp_hid_dim': 5120,
		'drop': 0.1,
		'n_classes': 1000
	}

device = 'cuda'
batch_size = 512
