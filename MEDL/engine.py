"""
Module to train (for one epoch) and evaluate the network.
"""

import torch
from torch import nn
from torch.autograd import Variable

from matcher import *


def train(model: nn.Module,
		  criterion_cls: nn.Module, criterion_loc: nn.Module,
		  matcher: HungarianMatcher, weight_cls: float,
		  optimizer: torch.optim.Optimizer,
		  data_sf, data_bh, label_cls, label_loc,
		  batch_size: int, nsr: float, mean_sf, std_sf, mean_bh, std_bh,
		  device: torch.device):
	"""
	Perform a training step over the entire training set (1 epoch of training).
	"""

	model.train()
	criterion_cls.train()
	criterion_loc.train()
	
	num_batches = 0
	running_loss_cls = 0.
	running_loss_loc = 0.
	running_loss = 0.

	n_data = data_sf.shape[0]
	n1_label = label_cls.shape[1]

	shuffled_indices = torch.randperm(n_data)

	# Check the learning rate of optimizer.
	# print(optimizer.state_dict()['param_groups'][0]['lr'])

	for count in range(0, n_data, batch_size):
		optimizer.zero_grad()

		# - FORWARD PASS -
		indices = shuffled_indices[count:count+batch_size]

		# data size: (n_data, nx, nt)
		# label_cls size: (n_data, n1_label)
		# label_loc size: (n_data, n2_label)

		# Create minibatch data of size (batch_size, 1, nx, nt), where 1 represents one channel.
		# surface data
		minibatch_data_sf = (data_sf[indices] + nsr * torch.randn(data_sf[indices].size())).to(device)
		# minibatch_data = data[indices].unsqueeze(dim=1).to(device)   # without Gaussian noise added

		minibatch_data_bh = (data_bh[indices] + nsr * torch.randn(data_bh[indices].size())).to(device)

		# Create minibatch labels for classification of size (batch_size, n1_label, 1)
		# and minibatch labels for location of size (batch_size, n1_label, 3).
		minibatch_label_cls = label_cls[indices].unsqueeze(dim=2).to(device)
		minibatch_label_loc = label_loc[indices].view(-1, n1_label, 3).to(device)

		inputs_sf = Variable((minibatch_data_sf - mean_sf) / std_sf, requires_grad=False)
		inputs_bh = Variable((minibatch_data_bh - mean_bh) / std_bh, requires_grad=False)
		labels_cls = Variable(minibatch_label_cls, requires_grad=False)
		labels_loc = Variable(minibatch_label_loc, requires_grad=False)

		outputs = model(inputs_sf, inputs_bh)

		# - BACKWARD PASS -
		idx_ = matcher(outputs, labels_cls, labels_loc)
		pred_idx = get_pred_permutation_idx(idx_)
		tgt_idx = get_tgt_permutation_idx(idx_)

		# Note: The following size values are only for this case.
		# predicted probabilities
		prob = outputs['pred_logits'].softmax(-1)[:, :, :-1]   # size of (batch_size, num_queries, 1)
		prob = prob[pred_idx]                                  # size of (batch_size * num_queries, 1)
		# predicted locations
		pred_loc = outputs['pred_locations']                   # size of (batch_size, num_queries, 2)
		pred_loc = pred_loc[pred_idx]                          # size of (batch_size * num_queries, 2)

		# target classes
		labels_cls = labels_cls[tgt_idx]                       # size of (batch_size * num_queries, 1)
		# target locations
		labels_loc = labels_loc[tgt_idx]                       # size of (batch_size * num_queries, 2)

		# classification loss term
		loss_cls = criterion_cls(prob, labels_cls)

		# location loss term
		keep = labels_cls.max(-1).values > 0.7   # True or False of size (batch_size * num_queries)
		loss_loc = criterion_loc(pred_loc[keep], labels_loc[keep])
		# For special case of zero-event input:
		if torch.count_nonzero(keep) == 0:
			loss_loc = torch.tensor(0.)

		# total loss term
		loss = weight_cls * loss_cls + (1. - weight_cls) * loss_loc

		loss.backward()
		# nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
		optimizer.step()

		# COMPUTE STATS
		num_batches += 1
		running_loss_cls += loss_cls.detach().item()
		running_loss_loc += loss_loc.detach().item()
		running_loss += loss.detach().item()

	running_loss_cls /= num_batches
	running_loss_loc /= num_batches
	running_loss /= num_batches

	return running_loss_cls, running_loss_loc, running_loss


"""
@torch.no_grad()
def evaluate(model: nn.Module,
			 criterion_cls: nn.Module, criterion_loc: nn.Module,
			 matcher: HungarianMatcher, weight_cls: float,
			 data, label_cls, label_loc,
			 nsr: float, mean, std,
			 device: torch.device):
	///
	Perform an evaluation step over the entire validation set.
	///

	model.eval()
	criterion_cls.eval()
	criterion_loc.eval()

	num_batches = 0
	running_loss_cls = 0.
	running_loss_loc = 0.
	running_loss = 0.

	n_data = data.shape[0]
	n1_label = label_cls.shape[1]

	# Define the batch size of data that we want to feed to the model.
	batch_size = n_data

	for count in range(0, n_data, batch_size):
		indices = range(count, count+batch_size)

		minibatch_data = (data[indices] + nsr * torch.randn(data[indices].size())).to(device)
		# minibatch_data = data[indices].unsqueeze(dim=1).to(device)   # without Gaussian noise added

		minibatch_label_cls = label_cls[indices].unsqueeze(dim=2).to(device)
		minibatch_label_loc = label_loc[indices].view(-1, n1_label, 3).to(device)

		inputs = Variable((minibatch_data - mean) / std, requires_grad=False)
		labels_cls = Variable(minibatch_label_cls, requires_grad=False)
		labels_loc = Variable(minibatch_label_loc, requires_grad=False)

		outputs = model(inputs)

		idx_ = matcher(outputs, labels_cls, labels_loc)
		pred_idx = get_pred_permutation_idx(idx_)
		tgt_idx = get_tgt_permutation_idx(idx_)

		# Note: The following size values are only for this case.
		# predicted probabilities
		prob = outputs['pred_logits'].softmax(-1)[:, :, :-1]   # size of (batch_size, num_queries, 1)
		prob = prob[pred_idx]                                  # size of (batch_size * num_queries, 1)
		# predicted locations
		pred_loc = outputs['pred_locations']                   # size of (batch_size, num_queries, 2)
		pred_loc = pred_loc[pred_idx]                          # size of (batch_size * num_queries, 2)

		# target classes
		labels_cls = labels_cls[tgt_idx]                       # size of (batch_size * num_queries, 1)
		# target locations
		labels_loc = labels_loc[tgt_idx]                       # size of (batch_size * num_queries, 2)

		# classification loss term
		loss_cls = criterion_cls(prob, labels_cls)

		# location loss term
		keep = labels_cls.max(-1).values > 0.7
		loss_loc = criterion_loc(pred_loc[keep], labels_loc[keep])
		# For special case of zero-event input:
		if torch.count_nonzero(keep) == 0:
			loss_loc = torch.tensor(0.)

		# total loss term
		loss = weight_cls * loss_cls + (1. - weight_cls) * loss_loc

		num_batches += 1
		running_loss_cls += loss_cls.detach().item()
		running_loss_loc += loss_loc.detach().item()
		running_loss += loss.detach().item()

	running_loss_cls /= num_batches
	running_loss_loc /= num_batches
	running_loss /= num_batches

	return running_loss_cls, running_loss_loc, running_loss
"""
