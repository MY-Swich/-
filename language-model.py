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
MAX_VOCAB_SIZE = 50
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
if USE_CUDA:
    model = model.cuda()

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5) #学习率降一半

VOCAB_SIZE = len(TEXT.vocab)
GRAD_CLIP = 5.
val_losses = []

def evaluate(model, data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(train_iter)
    print(len(train_iter))
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad = False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_loss = loss.item() * np.multiply(*data.size())
            total_count = np.multiply(*data.size())
    loss = total_loss / total_count
    model.train()
    return loss

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

        print(i, loss.item())
        if i % 3 == 0:
            print("loss ", loss.item())

        if i % 5 == 0:
            # torch.save(model.state_dict(), "lm.pth")

            val_loss = evaluate(model, val_iter)
            if len(val_losses) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), "lm.pth")
                print("beat model saved to lm.pth")
            else:
                scheduler.step()
            val_losses.append(val_loss)

#加载
best_model = RNNModel(vocab_size=len(TEXT.vocab),
                      embed_size=EMBEDDING_SIZE,
                      hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    best_model = best_model.cuda()
best_model.load_state_dict(torch.load("lm.pth"))

#使用训练好的模型生成一些句子。
hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    word_weights = output.squeeze().exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))