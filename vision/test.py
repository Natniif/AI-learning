import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Resize((227,227)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# already downloaded
train = torchvision.datasets.CIFAR10(root='./data/CIFAR10/train_CIFAR10', train=True,
                                    download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64,
                                          shuffle=True, num_workers=2)
# already downloaded 
test = torchvision.datasets.CIFAR10(root='./data/CIFAR10/test_CIFAR10', train=False,
                                    download=False, transform=transform)

test_loader = torch.utils.data.DataLoader(train, batch_size=64,
                                          shuffle=True, num_workers=2)

	
class AlexNet(nn.Module): 
    def __init__(self, num_classes=10): 
        super(AlexNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.LocalResponseNorm(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.l2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), 
            nn.LocalResponseNorm(384),
            nn.ReLU())
        
        self.l4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU())
        
        self.l5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=9216, out_features=4096), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=4096, out_features=num_classes))
        
    def forward(self, x): 
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out) 
        return out
    
torch.manual_seed(423984) 
alexnet = AlexNet(10)
alexnet.to(device)

loss_fn = nn.CrossEntropyLoss()
# optimizer parameters according to paper
optimizer = torch.optim.SGD(alexnet.parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)

from tqdm.auto import tqdm 

torch.manual_seed(424)

epochs = 5

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n-----')
    
    train_loss = 0
    
    for batch, (x, y) in enumerate(train_loader): 
        x = x.to(device)
        
        y_pred = alexnet(x)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train)
    
    test_loss = 0
    
    # with torch.inference_mode(): 
    #     for x, y in test:
    #         test_pred = alexnet(x) 
            
    #         loss = loss_fn(test_pred, y)
    #         test_loss += loss
            
    #     test_loss /= len(test)
    
    print(f"\nTrain loss: {train_loss:.5f}\n")

