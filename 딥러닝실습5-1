import torch
import torch.nn as nn
import tqdm
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor,Resize,RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.models.vgg import vgg16

device = 'cuda' if torch.cuda.is_available() else "cpu"

transforms = Compose([
    Resize(224),
    RandomCrop((224, 224), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465),
              std=(0.247, 0.243, 0.261))
])

model = vgg16(pretrained=True)
fc = nn.Sequential(
    nn.Linear(512*7*7, 4096),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 10)
)

model.classifier = fc
model.to(device)

training_data = CIFAR10(root='./', train=True,
                        download=True, transform=transforms)
test_data = CIFAR10(root='./', train=False,
                    download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

lr = 1e-4
optim = Adam(model.parameters(),lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    #tqdm 반복문으로 찍히는 값을 이쁘게 보여주는 역할
    for data, label in iterator:
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
