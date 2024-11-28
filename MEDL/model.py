import torch
import torch.nn as nn


class DETRnet(nn.Module):
	"""
	DETR architecture-based network for Microseismic Event Detection and Location.
	"""

	def __init__(self, num_classes, hidden_dim=64, nheads=8, num_encoder_layers=4, num_decoder_layers=4):
		super().__init__()

		# CNN BACKBONE to extract a compact feature representation
  
		# !cnn block for surface data!
		self.backbone_sf_conv1 = nn.Conv2d(6,   32,  kernel_size=3, stride=1, padding=1)
		self.backbone_sf_conv2 = nn.Conv2d(32,  128, kernel_size=3, stride=1, padding=1)
		self.backbone_sf_conv3 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
		self.backbone_sf_norm1 = nn.BatchNorm2d(32)
		self.backbone_sf_norm2 = nn.BatchNorm2d(128)
		self.backbone_sf_norm3 = nn.BatchNorm2d(512)
		self.backbone_sf_activate = nn.ReLU()
		self.backbone_sf_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

		# !cnn block for borehole data!
		self.backbone_bh_conv1 = nn.Conv2d(3,   32,  kernel_size=3, stride=1, padding=1)
		self.backbone_bh_conv2 = nn.Conv2d(32,  128, kernel_size=3, stride=1, padding=1)
		self.backbone_bh_conv3 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
		self.backbone_bh_norm1 = nn.BatchNorm2d(32)
		self.backbone_bh_norm2 = nn.BatchNorm2d(128)
		self.backbone_bh_norm3 = nn.BatchNorm2d(512)
		self.backbone_bh_activate = nn.ReLU()
		self.backbone_bh_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.backbone_bh_pool_tdim = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0)

		# CONVERSION LAYER to reduce the channel dimension
		self.conversion = nn.Conv2d(1024, hidden_dim, kernel_size=1)

		# TRANSFORMER
		self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, activation='gelu')

		# FFN to decode the N output embeddings from Transformer into class and location predictions
		self.linear_class = nn.Linear(hidden_dim, num_classes + 1)      # extra 1 for no_event class
		self.linear_location = nn.Linear(hidden_dim, 3)                 # X, Y and Z source locations
  
		# Spatial Positional Encodings for feature maps
		self.row_embed = nn.Parameter(torch.rand(10, hidden_dim // 2))
		self.col_embed = nn.Parameter(torch.rand(10, hidden_dim // 2))

		# Object Queries as the input to Transformer decoder
		self.query_pos = nn.Parameter(torch.rand(5, hidden_dim))


	def forward(self, inputs_surface, inputs_borehole):
		batch_size = inputs_surface.shape[0]

		# Propagate inputs through the CNN BACKBONE.
		# cnn for surface
		x_sf = self.backbone_sf_conv1(inputs_surface)
		x_sf = self.backbone_sf_norm1(x_sf)
		x_sf = self.backbone_sf_activate(x_sf)
		x_sf = self.backbone_sf_pool(x_sf)

		x_sf = self.backbone_sf_conv2(x_sf)
		x_sf = self.backbone_sf_norm2(x_sf)
		x_sf = self.backbone_sf_activate(x_sf)
		x_sf = self.backbone_sf_pool(x_sf)

		x_sf = self.backbone_sf_conv3(x_sf)
		x_sf = self.backbone_sf_norm3(x_sf)
		x_sf = self.backbone_sf_activate(x_sf)
		x_sf = self.backbone_sf_pool(x_sf)

		# cnn for borehole
		x_bh = self.backbone_bh_conv1(inputs_borehole)
		x_bh = self.backbone_bh_norm1(x_bh)
		x_bh = self.backbone_bh_activate(x_bh)
		x_bh = self.backbone_bh_pool(x_bh)

		x_bh = self.backbone_bh_conv2(x_bh)
		x_bh = self.backbone_bh_norm2(x_bh)
		x_bh = self.backbone_bh_activate(x_bh)
		x_bh = self.backbone_bh_pool(x_bh)

		x_bh = self.backbone_bh_conv3(x_bh)
		x_bh = self.backbone_bh_norm3(x_bh)
		x_bh = self.backbone_bh_activate(x_bh)
		x_bh = self.backbone_bh_pool_tdim(x_bh)

		x = torch.cat([x_sf, x_bh], dim=1)

		# Convert from 512 to hidden_dim feature planes.
		h = self.conversion(x)
		# h of size (batch_size, hidden_dim, H, W)

		# Construct positional encodings for feature planes.
		H, W = h.shape[-2:]
		pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
						 self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1)
		pos = pos.flatten(0, 1).unsqueeze(1).repeat(1, batch_size, 1)
		# pos of size (H * W, batch_size, hidden_dim)

		# Propagate through the TRANSFORMER.
		h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
							 self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)).transpose(0, 1)
		# encoder input of size (H * W, batch_size, hidden_dim)
		# decoder input of size (num_queries, batch_size, hidden_dim)
		# transformer output of size: the same as decoder input size
		# h of size (batch_size, num_queries, hidden_dim)

		# Project TRANSFORMER outputs to class and location predictions through FFN.
		return {'pred_logits': self.linear_class(h), 'pred_locations': self.linear_location(h).sigmoid()}
		# outputs["pred_logits"] size: (batch_size, num_queries, num_classes + 1)
		# outputs["pred_locations"] size: (batch_size, num_queries, 2)
