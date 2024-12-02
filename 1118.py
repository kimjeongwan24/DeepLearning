import torch
import torch.nn as nn
#nn neural network을 구성하는 module들이 들어있음
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Hpyer-parameters
input_size = 28*28
output_size = 10
num_epochs = 300
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)
# Data loader = (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
#
# x_train = np.array([[3.3],[4.4],[5.5],[6.6]],
#                    dtype=np.float32)
#
# y_train = np.array([[1.7],[2.76],[2.09],[3.19]],
#                    dtype=np.float32)


# Logistic regression model
model = nn.Linear(input_size, output_size)

#criterion = nn.MSELoss()

# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Tain the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1,input_size)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #inputs = torch.from_numpy(x_train)
    #targets = torch.from_numpy(y_train)

    # outputs = model(inputs)
    # loss = criterion(outputs,targets)
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    if (epoch+1)%5 == 0:
        print('Epoch [{}/{}], Loss {:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outptus = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print("acc {}%".format(100*correct/total))

# predicted = model(torch.from_numpy(x_train)).detach().numpy()
# plt.plot(x_train, y_train, 'ro', label='Original data')
# plt.plot(x_train, predicted, label='Fitted Line')
# plt.legend()
# plt.show()

