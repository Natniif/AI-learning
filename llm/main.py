import os
from dataclasses import dataclass
from argparse import ArgumentParser
from llm_models import GPT


import torch 
import torch.nn as nn 
import torch.functional as F

import numpy as np

import torchtext
from torchtext.data.utils import get_tokenizer

# ------------ SETTINGS ------------

WDIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config(): 
	num_batches = 8
	num_folds = 12
	num_heads = 4
	epochs = 4

# ------------ DATALOADER ------------
def load_data(filename, split): 
	file_dir = os.path.join(WDIR, "/data", filename + ".txt")
	text = []
	with open(file_dir, "r") as F: 
		pass



# ------------ TRAIN ------------
def train():
	pass




# ------------ TEST ------------
def test():
	pass


# ------------ VAL ------------
def val():
	pass



# ------------ INFER ------------
def infer():
	pass



# ------------ MAIN ------------
if __name__ == '__main__': 
	pass



