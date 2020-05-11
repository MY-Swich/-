import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from UNet.unet import unet
from UNet.dataset import LiverDataset

#是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
        return model




def train(batch_size):
    model = unet(3, 1).to(device)
    #损失函数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters()) #默认学习率
    liver_dataset = LiverDataset('./data/train', transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(
        liver_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    train_model(model, criterion, optimizer, dataloaders)

def Test(path):
    model = unet(3, 1)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            print((img_y))
            plt.imshow(img_y)
            plt.pause(0.01)
        # plt.savefig("filename.png")
        plt.show()

batch_size = 8
ckpt = 'weights_19.pth' #weights_19.pth路径
train(batch_size)
Test(ckpt)