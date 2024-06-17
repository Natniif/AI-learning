import torch 
import torch.nn as nn

"""
Paper:	https://arxiv.org/pdf/1506.02640
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class YOLOv3(nn.Module):
	def __init__(self, num_classes):
		super(YOLOv3, self).__init__()

		self.layers = nn.ModuleList()
		self.layers.append(ConvBlock(3, 32, 3, 1, 1))
		self.layers.append(ConvBlock(32, 64, 3, 2, 1))
		self.layers.append(ConvBlock(64, 128, 3, 2, 1))
		self.layers.append(ConvBlock(128, 256, 3, 2, 1))
		self.layers.append(ConvBlock(256, 512, 3, 2, 1))
		self.layers.append(ConvBlock(512, 1024, 3, 2, 1))

		self.conv_final = nn.Conv2d(1024, (num_classes + 5) * 3, 1, 1, 0)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		x = self.conv_final(x)
		# outputs tensor box
		return x.view(x.size(0), -1, x.size(2), x.size(3))



if __name__ == "__main__": 
	pass