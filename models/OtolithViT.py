import torch.nn as nn
import timm

class OtolithViT(nn.Module):
	def __init__(self, model_name, out_features=1, train_indices=None, val_indices=None):
		super(OtolithViT, self).__init__()

		self.model_name = model_name

		self.train_indices = train_indices
		self.val_indices = val_indices

		vit_cfg = model_name.split('_')
		patch = vit_cfg[1]
		resolution = vit_cfg[2]
		clip = ''
		if vit_cfg[-1] == 'clip':
			clip = '_clip'

		# Loading desired ViT architecture from timm.
		timm_vit_cfg = f'vit_base_patch{patch}{clip}_{resolution}'
		self.vit = timm.create_model(
			model_name=timm_vit_cfg,
			pretrained=True,
			dynamic_img_size=True
		)
		
		# Reshaping final linear layer to have desired output size, i.e., 1 for regression, number of classes for classification.
		self.vit.reset_classifier(num_classes=out_features)
	
	def forward(self, x):
		return self.vit(x).squeeze()

	def unfreeze(self, layers, learning_rates):
		# Unfreeze (enable gradient computation) given layers with given learning rates, making them available to be optimized.
		to_unfreeze = []
		for layer_string, learning_rate in zip(layers, learning_rates):
			to_append = None
			if layer_string.startswith('blocks'):
				layer = getattr(self.vit.blocks, layer_string.split('.')[1])
			else:
				layer = getattr(self.vit, layer_string)
			
			layer.requires_grad_(True)

			if layer_string == 'cls_token' or layer_string == 'pos_embed':
				to_append = layer
			else:
				to_append = layer.parameters()

			to_unfreeze.append(
				{
					'params': to_append,
					'lr': learning_rate,
				}
			)

		return to_unfreeze
		
	def freeze(self, layers):
		# Freeze (disable gradient computation) given layers, making them unable to be optimized.
		for layer_string in layers:
			if layer_string.startswith('blocks'):
				layer = getattr(self.vit.blocks, layer_string.split('.')[1])
			else:
				layer = getattr(self.vit, layer_string)

			layer.requires_grad_(False)