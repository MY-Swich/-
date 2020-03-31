import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
MAX_VOCAB_SIZE = 50000
NUM_EPOCHS = 2


TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="./data",
    train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=32, repeat=False, shuffle=True)

it = iter(train_iter)
batch = next(it)
# print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
# print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,1].data]))

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        ''' 该模型包含以下几层:
                    - 词嵌入层
                    - 一个循环神经网络层(RNN, LSTM, GRU)
                    - 一个线性层，从hidden state到输出单词表
                    - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, text, hidden):
        emb = self.embed(text)
        output, hidden = self.lstm(emb, hidden)
        out_vocab = self.linear(output.view(-1, output.shape[2]))
        out_vocab = out_vocab.view(output.size(0), output.size(1), out_vocab.size(-1))
        return out_vocab, hidden

    def init_hidden(self, bsz, requires_grad = True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True),
                weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True))


model = RNNModel(vocab_size=len(TEXT.vocab),
                 embed_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

VOCAB_SIZE = len(TEXT.vocab)
GRAD_CLIP = 5.

for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)

        loss = loss_fn(output.view(-1, VOCAB_SIZE),target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        print(i,loss.item())
        if i % 100 == 0:
            print("loss ",loss.item())
