import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as Data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 32 x 32
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #激活 28*28
        x = F.max_pool2d(x, 2, 2) #池化 14*14
        x = F.relu(self.conv2(x)) #10*10
        x = F.max_pool2d(x, 2, 2) #5*5
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

BATCH_SIZE=10

model = Net()
input = torch.randn(1000,1,32,32)

target = torch.randn(1000)

print(input)

torch_dataset = Data.TensorDataset(input, target)

loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True                # random shuffle for training
)

for step, (batch_x, batch_y) in enumerate(loader):
    print('| Step: ', step, '| batch x: ',batch_x.size(), '| batch y: ', batch_y.size())


#
#
lr = 0.1
momentum = 0.5
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)
# batch_size=32
#
# inp = torch.utils.data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     pin_memory=True
# )
#
#
def train(model, inp, optimizer):
    model.train()
    for i, (out, tag) in enumerate(inp):
        output = model(out)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, tag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss: {}".format(loss.item()))




train(model, loader , optimizer)