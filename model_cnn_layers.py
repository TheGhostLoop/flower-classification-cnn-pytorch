import torch.nn as nn
class Flower_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )

    self.fc = nn.Linear(64*27*27,14)
  
  def forward(self,x):
    x = self.conv(x)
    x = x.view(x.size(0),-1)
    x = self.fc(x)
    return x
