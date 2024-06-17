import torch 
import torch.nn as nn

class YOLO(nn.Module): 
	def __init__(self, num_features, pretrained=False): 
		super(YOLO, self).__init__()

		self.conv1 = nn.Conv2d(64, 7, )
		self.conv1 = 
		self.conv1 = 
		self.conv1 = 
		self.conv1 = 
		self.conv1 = 
		self.conv1 = 
		self.conv1 = 

		self.max_pool = self.MaxPool(kernel_size=2, stride=2)
		self.l1 = nn.Linear(1024, 4096)
		self.l2 = nn.Linear(4096, 7)


	def forward(self, x): 
		pass


if __name__ == "__main__": 
	pass