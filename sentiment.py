import torch
from torchtext import data
from torchtext import datasets
import spacy
import torch.nn as nn
import random
import torch.nn.functional as F

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

#网速太慢，下不到资源，先注释掉，这个是随机分割
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super(WordAVGModel,self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.linear = nn.Linear(embedding_size, output_size)


    def forward(self, text):
        embedded = self.embed(text)  # [seq_len,  batch_size, embedding_size]
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze()
        return self.linear(pooled)

VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi(TEXT.pad_token)

model = WordAVGModel(vocab_size=VOCAB_SIZE,
                     embedding_size=EMBEDDING_SIZE,
                     output_size=OUTPUT_SIZE,
                     pad_idx=PAD_IDX)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)

pretrained_embeddings = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, crit):

    epoch_loss, epoch_acc = 0.,0.
    model.train()
    total_len = 0
    for batch in iterator:
        preds = model(batch.text).squeeze()
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)

    return epoch_loss / len(iterator), epoch_acc / total_len

def evaluate(model, iterator, crit):

    epoch_loss, epoch_acc = 0.,0.
    model.evel()
    total_len = 0
    for batch in iterator:
        preds = model(batch.text).squeeze()
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)


        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)
    model.train()

    return epoch_loss / len(iterator), epoch_acc / total_len

N_EPOCHS = 10
best_valid_acc = 0.
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), "wordavg-model.pyh")

    print("Epoch", epoch, "Train Loss", train_loss, "Train Acc", train_acc)
    print("Epoch", epoch, "Valid Loss", valid_loss, "Valid Acc", valid_acc)


