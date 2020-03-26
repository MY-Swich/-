# FizzBuzz
#
# FizzBuzz是一个简单的小游戏。游戏规则如下：从1开始往上数数，当遇到3的倍数的时候，说fizz，当遇到5的倍数，说buzz，当遇到15的倍数，就说fizzbuzz，其他情况下则正常数数。
#
# 我们可以写一个简单的小程序来决定要返回正常数值还是fizz, buzz 或者 fizzbuzz。

import numpy as np
import torch

NUM_DIGITS = 10

#游戏规则处理
def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i % 5 == 0: return 2
    elif i % 5 == 0: return 1
    else: return 0

def fizz_buzz_decode(i, prediction):
    return [str(i), "fize", "buzz", "fizzbuzz"][prediction]

def helper(i):
    print(fizz_buzz_decode(i, fizz_buzz_encode(i)))

# for i in range(1,16):
#     helper(i)


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])

#输入输出数据
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# 构建网络
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)

#loss函数，优化函数
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

#训练
BATCH_SIZE = 128
for epoch in range(20000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        # 前向传播
        y_pred = model(batchX)

        # 获得loss
        loss = loss_fn(y_pred,batchY)

        print("Epoch:",epoch, loss.item())

        #后向传播
        optimizer.zero_grad()
        loss.backward()
        #计算梯度
        optimizer.step()

testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)

predictions = zip(range(1, 101), testY.max(1)[1].data.tolist())
print([fizz_buzz_decode(i, x) for i, x in predictions])