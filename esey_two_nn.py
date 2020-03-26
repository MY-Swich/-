import torch
import torch.nn as nn

#N为批次数，D_in输入，H隐藏层，D_out输出层
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

class TwolayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwolayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TwolayerNet(D_in, H, D_out)

loss_fn = nn.MSELoss(reduction='sum')
learn_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)

for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
