import torch 
import torch.nn as nn
import torchvision
from torchvision.models import  resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

'''
From paper: https://arxiv.org/pdf/1512.03385
'''

class Block18(nn.Module): 
	def __init__(self, in_channels, out_channels, stride=1): 
		super(Block18, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.downsample = None
		if stride != 1 or in_channels != out_channels:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
		    )

	def forward(self, x): 
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class Block50(nn.Module): 
	expansion = 4
	def __init__(self, in_channels, out_channels, stride=1): 
		super(Block50, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)	
		self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

		self.relu = nn.ReLU(inplace=True)

		self.downsample = None
		# residual layer
		if stride != 1 or in_channels != out_channels * self.expansion:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels * self.expansion),
		    )

	def forward(self, x): 
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out	


class ResNet(nn.Module): 
	models = {
		'18' : resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
		'34' : resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
		'50' : resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
		'101' : resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
		'152' : resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
	}

	num_blocks = {
		'18' : [2, 2, 2, 2],
		'34' : [3, 4, 6, 3],
		'50' : [3, 4, 6, 3],
		'101' : [3, 4, 23, 3],
		'152' : [3, 8, 36, 3]
	}
	def __init__(self, size, num_classes, pretrained=True): 
		super(ResNet, self).__init__()
		self.size = size
		self.pretrained = pretrained

		if pretrained: 
			self.resnet = self.models[size]
			self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
			num_features = self.resnet.fc.in_features
			self.resnet.fc = nn.Linear(num_features, num_classes)

		else:
			self.in_channels = 64
			self.layer1 = nn.Sequential(
			    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
			    nn.BatchNorm2d(64),
			    nn.ReLU(inplace=True),
			    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
			)
			self.layer2 = self._make_layer(64, self.num_blocks[self.size][0])
			self.layer3 = self._make_layer(128, self.num_blocks[self.size][1], stride=2)
			self.layer4 = self._make_layer(256, self.num_blocks[self.size][2], stride=2)
			self.layer5 = self._make_layer(512, self.num_blocks[self.size][3], stride=2)
			self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

			if int(self.size) >= 50:
				self.fc = nn.Linear(512 * Block50.expansion, num_classes)
			else: 
				self.fc = nn.Linear(512, num_classes)

	def _make_layer(self, out_channels, blocks, stride=1): 
		layers = []
		if int(self.size) >= 50:
			layers.append(Block50(self.in_channels, out_channels, stride))
			self.in_channels = out_channels * Block50.expansion
			for _ in range(1, blocks):
				layers.append(Block50(self.in_channels, out_channels))
			return nn.Sequential(*layers)
		else: 
			layers.append(Block18(self.in_channels, out_channels, stride))
			self.in_channels = out_channels
			for _ in range(1, blocks):
				layers.append(Block18(out_channels, out_channels))
			return nn.Sequential(*layers)


	def forward(self, x): 
		if self.pretrained:
			return self.resnet(x)
		else:
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)
			x = self.layer5(x)
			x = self.avg_pool(x)
			x = torch.flatten(x, 1)
			x = self.fc(x)
			return x

if __name__ == "__main__": 
	# Verify the model with a dummy input
	res50 = ResNet(size='50', num_classes = 10)
	res18 = ResNet(size='18', num_classes = 10)
	res18_imp = ResNet(size='18', num_classes=10, pretrained=False)
	res50_imp = ResNet(size='50', num_classes=10, pretrained=False)

	dummy_input = torch.randn(1, 3, 227, 227)  # Batch size of 1, 3 color channels, 227x227 image size
	output_res50 = res50(dummy_input)
	output_res18 = res18(dummy_input)
	output_res18_imp = res18_imp(dummy_input)
	output_res50_imp = res50_imp(dummy_input)

	print("Output shape ResNet50:", output_res50.shape)  # Should match (1, 10) for 10 classes
	print("Output shape ResNet18:", output_res18.shape)  # Should match (1, 10) for 10 classes
	print("Output shape ResNet18 Self Implemented:", output_res18_imp.shape)  # Should match (1, 10) for 10 classes
	print("Output shape ResNet18 Self Implemented:", output_res50_imp.shape)  # Should match (1, 10) for 10 classes