import numpy as np
import torch
import random


def set_seed(seed):
	"""
	Set all random seeds to a fixed value and take out any randomness from cuda kernels.
	"""

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled   = False

	return True


def avail_device():
	"""
	Check the available device (CPU or GPU).
	"""

	device = 'cpu'

	if torch.cuda.device_count() > 0 and torch.cuda.is_available():
		device = 'cuda'
		# print("GPU is working hard for you!")
	else:
		print("No GPU available!")

	return device