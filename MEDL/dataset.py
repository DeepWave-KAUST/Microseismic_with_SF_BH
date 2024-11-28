import numpy as np
import torch


def read_data(file_name, n_data, nx, nt):
	"""
	Read seismic data.

	Params:
		file_name: name of file that stores the seismic data
		n_data: number of data samples
		nx: number of space samples for each data
		nt: number of time samples for each data
	"""

	with open(file_name, 'rb') as f:
		data_ = np.fromfile(f, np.single)
	# original data size: (nt * nx * n_data)
	data = torch.from_numpy(data_)
	# to size: (n_data, nx, nt)
	data = data.view(nt, nx, n_data).transpose(0, 2).float()

	return data


def read_label(file_name, n_data, n1_label, n2_label):
	"""
	Read corresponding labels.

	Params:
		file_name: name of file that stores the corresponding labels
		n_data: number of data samples
		n1_label: length of class label for each data
		n2_label: length of location label for each data
	"""

	n_label = n1_label + n2_label
	with open(file_name, "rb") as f:
		label_ = np.fromfile(f, np.single)
	# original label size: (n_data * n_label)
	label = torch.from_numpy(label_)
	# to size: (n_data, n_label)
	label = label.view(n_data, n_label)
	# part1 - class labels
	label_cls = label[:, :n1_label].float()
	# part2 - location labels
	label_loc = label[:, n1_label:].float()

	return label_cls, label_loc


def norm_label_loc(label_loc):
	"""
	Normalize location labels.

	Params:
		label_loc: location labels of size (n_data, n2_label)

	Note: Only support this specific case!
	"""
	
	# Source Area: [2.30, 3.10] in x-direction & [1.85, 2.65] in y-direction & [1.00, 2.60] in z-direction
	x0 = 2.30
	y0 = 1.85
	z0 = 1.75
	lx = 0.8
	ly = 0.8
	lz = 0.5
	i = np.arange(0, 15, 3)     # X locations are in *1 columns.
	j = np.arange(1, 15, 3)     # Y locations are in *2 columns.
	k = np.arange(2, 15, 3)     # Z locations are in *3 columns.
	label_loc[:, i] = (label_loc[:, i] - x0) / lx
	label_loc[:, j] = (label_loc[:, j] - y0) / ly
	label_loc[:, k] = (label_loc[:, k] - z0) / lz

	return label_loc
