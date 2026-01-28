from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.456,0.430,0.410],
        std = [0.240,0.229,0.225]
    )
])

train_dataset = ImageFolder(
    root="data/train",
    transform=transform
)
test_dataset = ImageFolder(
    root="data/val",
    transform=transform
)

train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=32)
test_loader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=32)


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


model = Flower_CNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 7
for epoch in range(epochs):
  model.train()
  epoch_loss=0
  for images,labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images)
    loss = criterion(logits,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()

  all_preds,all_labels = [],[]
  if epoch%1==0:
    for images,labels in train_loader:
      images = images.to(device)
      labels = labels.to(device)
      logits = model(images)
      preds = torch.argmax(logits,dim=1)
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels,all_preds)
    print(f"Epoch, {epoch+1}/{epochs} loss:{epoch_loss:.4f}, Accuracy score:: {acc:.4f}")
      

model.eval()
with torch.no_grad():
    all_preds,all_labels = [],[]
    for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits,dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"Final Accuracy Verdict:: {accuracy_score(all_labels,all_preds):.4f}")


