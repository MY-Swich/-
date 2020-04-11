import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, num_filters, filter_sizes, dropout):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters,
                      kernel_size=(fs, embedding_size))
            for fs in filter_sizes
        ])
        self.embed = nn.Embedding(vocab_size,embedding_size,padding_idx=pad_idx)
        # self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size))
        self.linear = nn.Linear(num_filters * len(num_filters), output_size)
        self.dropout = nn.Dropout(dropout)


    def forword(self, text):
        text = text.permute(1, 0) #[batch_size, seq_len]
        embedded = self.embed(text) #[batch_size, seq_len, embedding_size]
        embedded = embedded.unsqueeze(1) #[batch_size, 1, seq_len, embedding_size]
        # conved = F.relu(self.conv(embedded)) #[batch_size, num_filters, seq_len-filter_size+1, 1]
        # conved = conved.squeeze(3)
        conved = [F.relu(self.conv(embedded)).squeeze(3) for conv in self.convs]

        #max over time pooling
        # pooled = F.max_pool1d(conved, conved.shape[2])  #[batch_size, num_filters ,1]
        # pooled = pooled.squeeze(2)
        pooled = [F.max_pool1d(conved, conved.shape[2]).squeeze(2) for conv in conved]
        pooled = torch.cat(pooled, dim=1)
        pooled = self.dropout(pooled)

        return self.linear(pooled)

model = CNN(vocab_size=VOCAB_SIZE,
            embedding_size=EMBEDDING_SIZE,
            output_size=OUTPUT_SIZE,
            pad_idx=PAD_IDX,
            num_filters=100,
            filter_sizes=2,
            dropout=0.5)