import torch 
import torch.nn as nn
import torchvision
from torchvision.models import  resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

'''
From paper: https://arxiv.org/pdf/1512.03385
'''

class ResNet50(nn.Module): 
	def __init__(self, num_classes, pretrained=False ): 
		super(ResNet50, self).__init__()

		self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
		self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

		num_features = self.resnet.fc.in_features
		self.resnet.fc = nn.Linear(num_features, num_classes)

	def forward(self, x): 
		return self.resnet(x)

class ResNet18(nn.Module): 
	def __init__(self, num_classes, pretrained=False ): 
		super(ResNet18, self).__init__()

		self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

		num_features = self.resnet.fc.in_features
		self.resnet.fc = nn.Linear(num_features, num_classes)

	def forward(self, x): 
		return self.resnet(x)



if __name__ == "__main__": 
    # Verify the model with a dummy input
    res50 = ResNet50(10)
    res18 = ResNet18(10)

    dummy_input = torch.randn(1, 3, 227, 227)  # Batch size of 1, 3 color channels, 227x227 image size
    output_res50 = res50(dummy_input)
    output_res18 = res18(dummy_input)
    print("Output shape ResNet50:", output_res50.shape)  # Should match (1, 10) for 10 classes
    print("Output shape ResNet18:", output_res18.shape)  # Should match (1, 10) for 10 classes