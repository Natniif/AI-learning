import os 
import sys
import time
import random 
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from vison_models import *

if torch.cuda.is_available(): 
	device_type = "cuda"
elif torch.backends.mps.is_available():
	device_type = "mps"
else: 
	device_type = "cpu"

torch.manual_seed(123)
script_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config(): 
	epochs = 3
	batch_size = 32
	save_epoch = 1
	accumulation_steps = 4

def setup_data(dataset):
	# Define transformations including normalization
	transform = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Resize((227, 227)), 
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
	])

	data_dir = os.path.join(script_dir, 'data')

	if dataset == 'CIFAR10':
		if "cifar-10-batches-py" not in os.listdir(data_dir):
			trainset = torchvision.datasets.CIFAR10(root=data_dir + "/CIFAR10/train_CIFAR10" , train=True, download=True, transform=transform)
			testset = torchvision.datasets.CIFAR10(root=data_dir + "/CIFAR10/test_CIFAR10", train=False, download=True, transform=transform)
		else:
			trainset = torchvision.datasets.CIFAR10(root=data_dir + "/CIFAR10/train_CIFAR10", train=True, download=False, transform=transform)
			testset = torchvision.datasets.CIFAR10(root=data_dir + "/CIFAR10/test_CIFAR10", train=False, download=False, transform=transform)

    # Create data loaders
	trainloader = DataLoader(trainset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
	testloader = DataLoader(testset, batch_size=Config.batch_size, shuffle=False, num_workers=2)

	return trainset, trainloader, testset, testloader


# -------- TRAIN LOOP --------

def save_checkpoint(state, filename): 
	if "weights" not in os.listdir(script_dir): 
		os.makedirs("weights")

	checkpoint_dir = os.path.join(script_dir, "weights", filename)
	torch.save(state, checkpoint_dir)

# TODO: Implement into train, test and infer calls
def load_checkpoint(filename, model, optimizer, infer_mode=False): 
	checkpoint_dir = os.path.join(script_dir, filename)
	checkpoint = torch.load(filename)
	model.load_state_dict(checkpoint['model_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	if infer_mode == False: 
		optimizer.load_stat_dict(checkpoint['optimizer_state_dict'])
		return model, loss, optimizer, epoch, 
	else: 
		return model, loss, epoch


def train(model, loss_fn, optimizer, trainset, trainloader, testset, testloader, use_compile=False): 

	model.to(device_type)
	if use_compile: 
		model = torch.compile(model)

	for epoch in range(Config.epochs):
		print(f"Epoch: {epoch} / {Config.epochs}\n-----")
		train_loss = 0

		model.train()
		for batch, (x, y) in enumerate(tqdm(trainloader)): 
			x = x.to(device_type)
			y = y.to(device_type)

			y_pred = model(x)
			loss = loss_fn(y_pred, y)
			loss.backward()
			if (batch + 1) % Config.accumulation_steps:
				optimizer.step()
				optimizer.zero_grad()

			train_loss += loss
			if batch % 10 == 0: 
				print(f"Train loss: {train_loss/batch}")

		train_loss /= len(trainset)

		model.eval()
		test_loss = 0
		with torch.no_grad(): 
			for x, y in testloader: 
				x = x.to(device_type)
				y = y.to(device_type)
				
				test_pred = model(x)
				loss = loss_fn(test_pred, y)
				test_loss += loss

		test_loss /= len(testset)

		print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}\n")

		if epoch % Config.save_epoch: 
			print("Saving checkpoint...")

			checkpoint = {
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss,
			}
			save_checkpoint(checkpoint, filename=f'checkpoint_epoch_{epoch}.pth.tar')

	print("Training Complete")


def save_model(model, model_path_name):
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    model_path = os.path.join(weights_dir, model_path_name + ".pth")
    
    torch.save(model.state_dict(), model_path)
	

# -------- TESTING --------
def test(testset, model, from_pretrained=False, model_pth="", use_compile=True):
	if from_pretrained: 
		if not os.path.exists(model_pth):
			raise FileNotFoundError(f"Model weights file '{model_pth}' not found.")
		model.load_state_dict(torch.load(model_pth, map_location=device_type))

	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for image, label in testset:
			image = image.unsqueeze(0)
			outputs = model(image)
			_, predicted = torch.max(outputs.data, 1)
			total += 1
			if predicted == label: 
				correct += 1

			# hack to stop iterating through all of test cause its too large
			if total == 100:
				break

	print(f'Accuracy of the network on the 1000 test images: {100 * correct / total:.2f}%')


# -------- INFERENCE --------
def infer(model, dataset, filename="", from_pretrained=False, load_weights=False): 

	if from_pretrained == True: 
		checkpoint_path = os.path.join(script_dir, "weights", filename)
		checkpoint = load_checkpoint(checkpoint_path, infer_mode=True)
		model = checkpoint[0]

	with torch.inference_mode():
		if load_weights == False: 
			idx = random.randint(0, len(dataset) - 1)
			image, lable = dataset[idx]
			classes = dataset.classes
			itos = {i:s for i, s in enumerate(classes)}

			image = image.unsqueeze(0)
			output = model(image)
			_, predicted = torch.max(output, 1)

			image = image.squeeze(0)
			image = image.permute(1, 2, 0)
			image = image * 0.5 + 0.5 
			
			plt.imshow(image.numpy())
			plt.title(f"Predicted: {itos[int(predicted)]}, Real: {itos[lable]}")
			plt.show()
			return f"Predicted: {itos[int(predicted)]}, Real: {itos[lable]}"
		else: 
			return("Not implemented yet")


# -------- MAIN --------
if __name__ == "__main__": 
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--model', '-m', type=str, required=True, help="Name of model")
	parser.add_argument('--dataset', '-d', type=str, required=False, default="cifar10", help="Dataset used")
	parser.add_argument('--pretrained', '-p', type=bool, required=False)
	parser.add_argument('--run', '-r', type=str, help="What type of run to do. e.g. test, train, infer")
	parser.add_argument('--save', '-s', type=str, required=False, help="Name of the model you want to save")
	args = parser.parse_args()

	loaders = setup_data('CIFAR10')

	if args.pretrained == True:
		# TODO: implement pretrained and non-pretrained into all models
		models = {
			'alexnet' : AlexNet(10, pretrained=True), 
			'googlenet' : GoogLeNet(10, pretrained=True),
			'resnet18' : ResNet(size='18',num_classes=10),
			'resnet50' : ResNet(size='50',num_classes=10)
		}
	else: 
		models = {
			'alexnet' : AlexNet(10), 
			'googlenet' : GoogLeNet(10),
			'resnet18' : ResNet(size='18',num_classes=10),
			'resnet50' : ResNet(size='50',num_classes=10)
		}

	model = models[args.model]

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)

	if args.run == "train":
		train(model, loss_fn, optimizer, loaders[0], loaders[1], loaders[2], loaders[3])
	elif args.run == "test": 
		test(loaders[2], model)
	elif args.run == "infer": 
		print(infer(model, loaders[2]))

	if args.save:
		save_model(model, args.save)