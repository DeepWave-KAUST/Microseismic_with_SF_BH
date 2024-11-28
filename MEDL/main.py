# ---------- Microseismic Event Detection and Location with DETR ---------- #
# ----------                Author: Yuanyuan Yang                ---------- #
# ----------                    Hakuna Matata!                   ---------- #

# ---------------------------------------------------------------------------
# ------ Import Libraries ------
import time
import torch
import argparse

from pumbaa import *
from dataset import *
from model import *
from engine import *
from matcher import *


# ---------------------------------------------------------------------------
device = avail_device()
print(f'Device: {device} \n')

#parser = argparse.ArgumentParser('Set arguments.', add_help=False)
#parser.add_argument('--device', default='cuda')
#args = parser.parse_args()
#device = torch.device('cuda')


# ---------------------------------------------------------------------------
# ------ Define Data-related Parameters ------
n_train = 600   # number of training samples
# n_valid = 875    # number of validation samples

nx_sf = 180          # number of space samples for each data
nx_bh = 36
nt = 81         # number of time samples for each data
n1_label = 5     # length of class label for each data
n2_label = 15    # length of location label for each data


# ---------------------------------------------------------------------------
# ------ Define Training Hyperparameters ------
bs = 8                         # batch size
lr = 0.16                      # initial learning rate
n_epoch = 2000                 # number of training epochs

criterion_cls = nn.BCELoss()   # the criterion for classification task
criterion_loc = nn.MSELoss()   # the criterion for location task

weight_cls = 0.1               # the weight for classification loss term in the loss function
weight_cost_class = 1          # relative weight of the classification loss in the matching cost
weight_cost_location = 9       # relative weight of the location loss in the matching cost

nsr = 0.1/3.                   # noise to signal ratio (random Gaussian noise)


# ---------------------------------------------------------------------------
# ------ Load Data ------
# - Training Set -
# !surface data!
tmp_train_data_sf = read_data("../data/training_data_SF_3C_npc50.bin", n_train, nx_sf, nt)
train_data_sf_np = np.zeros((n_train, 6, 30, nt), dtype=np.single)
train_data_sf = torch.from_numpy(train_data_sf_np)
train_data_sf[:, 0:1, :, :] = tmp_train_data_sf[:, 0:30, :].unsqueeze(dim=1)
train_data_sf[:, 1:2, :, :] = tmp_train_data_sf[:, 30:60, :].unsqueeze(dim=1)
train_data_sf[:, 2:3, :, :] = tmp_train_data_sf[:, 60:90, :].unsqueeze(dim=1)
train_data_sf[:, 3:4, :, :] = tmp_train_data_sf[:, 90:120, :].unsqueeze(dim=1)
train_data_sf[:, 4:5, :, :] = tmp_train_data_sf[:, 120:150, :].unsqueeze(dim=1)
train_data_sf[:, 5:6, :, :] = tmp_train_data_sf[:, 150:180, :].unsqueeze(dim=1)

# !borehole data!
tmp_train_data_bh = read_data("../data/training_data_BH_3C.bin", n_train, nx_bh, nt)
train_data_bh_np = np.zeros((n_train, 3, 12, nt), dtype=np.single)
train_data_bh = torch.from_numpy(train_data_bh_np)
train_data_bh[:, 0:1, :, :] = tmp_train_data_bh[:, 0:12, :].unsqueeze(dim=1)
train_data_bh[:, 1:2, :, :] = tmp_train_data_bh[:, 12:24, :].unsqueeze(dim=1)
train_data_bh[:, 2:3, :, :] = tmp_train_data_bh[:, 24:36, :].unsqueeze(dim=1)

train_label_cls, train_label_loc = read_label("../data/training_label.bin", n_train, n1_label, n2_label)
train_label_loc = norm_label_loc(train_label_loc)

# # - Validation Set -
# valid_data = read_data("./data/validation_data.bin", n_valid, nx, nt)
# valid_label_cls, valid_label_loc = read_label("./data/validation_label.bin", n_valid, n1_label, n2_label)
# valid_label_loc = norm_label_loc(valid_label_loc)

# print(f'Number of Training   Samples: {train_data.numpy().shape[0]}')
# print(f'Number of Validation Samples: {valid_data.numpy().shape[0]}')


# ---------------------------------------------------------------------------
# ------ Characterize Data ------
# To check the data distribution and also for data normalization.
train_mean_sf = train_data_sf.mean()
# train_std  = train_data.std()
train_std_sf  = ((train_data_sf.std())**2 + nsr**2)**0.5

train_mean_bh = train_data_bh.mean()
train_std_bh  = ((train_data_bh.std())**2 + nsr**2)**0.5

# valid_mean = valid_data.mean()
# # valid_std  = valid_data.std()
# valid_std  = ((valid_data.std())**2 + nsr**2)**0.5


# ---------------------------------------------------------------------------
# ------ Build Network ------
set_seed(3407)
network = DETRnet(num_classes=1, hidden_dim=64, nheads=8, num_encoder_layers=4, num_decoder_layers=4)
# network.load_state_dict(torch.load('../Trained_Network.pt'))


# ---------------------------------------------------------------------------
# ------ Build Matcher ------
matcher = build_matcher(weight_cost_class=weight_cost_class, weight_cost_location=weight_cost_location)


# ---------------------------------------------------------------------------
# ------ Train Network ------
# - Information Sent to GPU -
network = network.to(device)
train_mean_sf = train_mean_sf.to(device)
train_std_sf  = train_std_sf.to(device)

train_mean_bh = train_mean_bh.to(device)
train_std_bh  = train_std_bh.to(device)
# valid_mean = valid_mean.to(device)
# valid_std  = valid_std.to(device)

print('Start training...')
tic = time.time()

# Define the optimizer and its scheduler.
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.25, last_epoch=-1)

# - Network Training -
for epoch in range(1, n_epoch+1):
	# Train one epoch over the entire training set.
	train_loss_cls, train_loss_loc, train_loss = train(network,
													   criterion_cls, criterion_loc,
													   matcher, weight_cls,
													   optimizer,
													   train_data_sf, train_data_bh, train_label_cls, train_label_loc,
													   bs, nsr, train_mean_sf, train_std_sf, train_mean_bh, train_std_bh,
													   device)

	# Write down the loss.
	if (epoch == 1) or (not epoch%5):
	#if epoch >= 1:
		print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_loss_cls, file=open("Training_Loss_cls.txt", "a"))
		print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_loss_loc, file=open("Training_Loss_loc.txt", "a"))
		print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, file=open("Training_Loss.txt", "a"))

		# valid_loss_cls, valid_loss_loc, valid_loss = evaluate(network,
		# 													  criterion_cls, criterion_loc,
		# 													  matcher, weight_cls,
		# 													  valid_data, valid_label_cls, valid_label_loc,
		# 													  nsr, valid_mean, valid_std,
		# 													  device)
		# print(epoch, valid_loss_cls, file=open("Validation_Loss_cls.txt", "a"))
		# print(epoch, valid_loss_loc, file=open("Validation_Loss_loc.txt", "a"))
		# print(epoch, valid_loss, file=open("Validation_Loss.txt", "a"))

	# Update the learning rate of optimizer.
	scheduler.step()

	# Save the network every 50 epochs.
	if not epoch%100:
		torch.save(network.state_dict(), f'../network/Trained_Network_ep{epoch}.pt')

# - Network Saving -
torch.save(network.state_dict(), '../network/Trained_Network.pt')


# ---------------------------------------------------------------------------
# ------ WELL DONE! ------
toc = time.time()
print(f'Training Time (hours): {((toc - tic) / 3600.):.2f}')
