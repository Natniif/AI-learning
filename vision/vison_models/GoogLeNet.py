import torch
import torch.nn as nn

import torchvision
from torchvision.models import googlenet, GoogLeNet_Weights


class Inception(nn.Module):
	def __init__(self, in_channels, out_1, red_3, out_3, red_5, out_5, out_pool):
		super(Inception, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_1, kernel_size=1)

		self.conv3 = nn.Sequential(
    		nn.Conv2d(in_channels, red_3, kernel_size=1, padding=0),
    		nn.Conv2d(red_3, out_3, kernel_size=3, padding=1))

		self.conv5 = nn.Sequential(
    		nn.Conv2d(in_channels, red_5, kernel_size=1),
    		nn.Conv2d(red_5, out_5, kernel_size=5, padding=2))

		self.max = nn.Sequential(
    		nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
    		nn.Conv2d(in_channels, out_pool, kernel_size=1))

	def forward(self, x):

		x1 = self.conv1(x)
		x2 = self.conv3(x)
		x3 = self.conv5(x)
		x4 = self.max(x)
		out = torch.cat([x1, x2, x3, x4], dim=1)
		return out


class GoogLeNet(nn.Module):
	def __init__(self, out_channels=10, pretrained=False):
		super().__init__()

		self.pretrained = pretrained

		if pretrained == True: 
			self.googlenet = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
			num_features = self.googlenet.fc.in_features
			self.linear = nn.Linear(num_features, out_channels)

		else:
			self.l1 = nn.Sequential(
				nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
				nn.MaxPool2d(kernel_size=3, stride=2))

			self.l2 = nn.Sequential(
				nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), 
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


			self.inc3a = Inception(192, 64, 96, 128, 16, 32, 32)
			self.inc3b = Inception(256, 128, 128, 192, 32, 96, 64)
			self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

			self.l4 = nn.Sequential(
				Inception(480, 192, 96, 208, 16, 48, 64), 
				Inception(512, 160, 112, 224, 24, 64, 64), 
				Inception(512, 128, 128, 256, 24, 64, 64), 
				Inception(512, 112, 144, 288, 32, 64, 64), 
				Inception(528, 256, 160, 320, 32, 128, 128),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) 

			self.inc5a = Inception(832, 256, 160, 320, 32, 128, 128)
			self.inc5b = Inception(832, 384, 192, 384, 48, 128, 128) 
			self.avg = nn.AvgPool2d(kernel_size=7, stride=1)

			self.drop = nn.Dropout(p=0.4)

			self.linear = nn.Linear(1024,out_channels)

	def forward(self, x): 

		if self.pretrained == True:
			x = self.googlenet(x)
			
		else: 
			x = self.l1(x) 
			x = self.l2(x)
			x = self.inc3a(x) 
			x = self.inc3b(x) 
			x = self.max3(x) 
			x = self.l4(x)
			x = self.inc5a(x) 
			print(x.shape)
			x = self.inc5b(x) 
			x = self.avg(x) 
			x = self.drop(x) 
			x = x.reshape(x.shape[0], -1)

		x = self.linear(x) 

		return x

if __name__ == "__main__": 
	dummy_input = torch.randn(1, 3, 227, 227)

	google_pretrained = GoogLeNet(10, pretrained=True)
	google = GoogLeNet(10)

	# TODO: get pretrained to work
	# out_google_pretrained = google_pretrained(dummy_input)
	out_google = google(dummy_input)
	# print("Output shape GoogLeNet with pretrained weights:", out_google_pretrained.shape)
	print("Output shape GoogLeNet:", out_google.shape)