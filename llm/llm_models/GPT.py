import torch 
import torch.nn as nn 
import torch.functional as F


class GPT(nn.Module):
	def __init__(self):
		super(GPT, self).__init__()