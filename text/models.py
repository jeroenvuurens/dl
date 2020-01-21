import torch
import torch.nn as nn
import torch.nn.functional as F

def single_layer_perceptron(input, output):

    class SingleLayerPerceptron(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.w1 = nn.Linear(input, output)

        def convert_input(self, x):
            return x[:,0,:,:].view(x.shape[0], -1)

        def forward(self, x):
            x = self.convert_input(x)
            x = self.w1(x)
            x = torch.log_softmax(x, dim=1)
            return x

    return SingleLayerPerceptron


class SentimentRNN1(nn.Module):
    def __init__(self, field, output_size):
        super().__init__()
        self.output_size = output_size
        self.vocab_size = len(field.vocab)
        self.embedding_dim = field.vocab.vectors.size(1)
        try:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).from_pretrained(field.vocab.vectors)
            print('using pre trained')
        except:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            print('using new')
        self.embedding.weight.requires_grad=False
        self.fc = nn.Linear(self.embedding_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def parameters(self):
        return iter([ p for p in super().parameters() if p.requires_grad ])

    def to(self, device):
        self.device = device
        super().to(device)

class SentimentRNN(nn.Module):
    def __init__(self, field, hidden_dim, output_size, n_layers=2, drop_prob=0.5, bidirectional=True):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.vocab_size = len(field.vocab)
        self.embedding_dim = field.vocab.vectors.size(1)
        try:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.embedding.weight.data.copy_(field.vocab.vectors)
        except:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            print('warning: not using pretrained embeddings')
        self.embedding.weight.requires_grad=False
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, n_layers, 
                            bidirectional=bidirectional, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

    def parameters(self):
        return iter([ p for p in super().parameters() if p.requires_grad ])

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        if self.bidirectional: # when bi-directional, we need the last two hidden layers
            x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            x = hidden[-1]
        x = self.dropout(x)
        out = self.fc(x)
        return out
   
    def to(self, device):
        self.device = device
        super().to(device)

def split_size(size):
    if type(size) is list or type(size) is tuple:
        return size
    return size, size

def compute_size(size, kernel, stride, padding, layer):
    s = (size - kernel + 2 * padding) / stride + 1
    assert s == int(s), f'size {size} at layer {layer} does not match with kernel size {kernel}, stride {stride} and padding {padding}'
    return int(s)

def log_softmax(x):
        return torch.log_softmax(x, dim=1)
