from torchvision.models import *
import torch
import torch.nn as nn

def single_layer_perceptron(input, output):

    class SingleLayerPerceptron(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(input, output)

        def forward(self, x):
            x = self.w1(x)
            pred_y = torch.sigmoid(x)
            return pred_y.view(-1)

        def predict(self, y):
            return torch.round(y)

    return SingleLayerPerceptron()

identity=lambda x:x

def flatten_r_image(x):
        return  x[:,0,:,:].view(x.shape[0], -1)

def log_softmax(x):
    return torch.log_softmax(x, dim=1)

def multi_layer_perceptron(*width, preprocess=identity, inner_activation=torch.relu, last_activation=torch.sigmoid, drop_prob=None):

    class MultiLayerPerceptron(nn.Module):
        def __init__(self):
            super().__init__()
            self.actions = [preprocess]
            for n, (i, o) in enumerate(zip(width[:-1], width[1:])):
                l = nn.Linear(i, o)
                self.actions.append(l)
                self.__setattr__(f'w{n+1}', l)
                if n < len(width) - 2:
                    if drop_prob is not None:
                        self.actions.append(nn.Dropout(p=drop_prob))
                        self.__setattr__(f'drop{n+1}', self.actions[-1])
                    self.actions.append(inner_activation)
                    self.__setattr__(f'activation{n+1}', self.actions[-1])
                elif last_activation != None:
                    self.actions.append(last_activation)
                    self.__setattr__(f'activation{n+1}', self.actions[-1])
            self.last_activation = last_activation
            if width[-1] == 1:
                self.reshape = (-1)
            else:
                self.reshape = (-1, width[-1])

        def forward(self, x):
            for a in self.actions:
                x = a(x)
            return x.view(self.reshape)

        def predict(self, y):
            if self.last_activation == torch.sigmoid:
                return torch.round(y)
            return y

    return MultiLayerPerceptron()

def two_layer_perceptron(input, hidden, output, a1=torch.relu):

    class TwoLayerPerceptron(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(input, hidden)
            self.w2 = nn.Linear(hidden, output)

        def forward(self, x):
            x = a1(self.w1(x))
            pred_y = torch.sigmoid(self.w2(x))
            return pred_y.view(-1)

        def predict(self, y):
            return torch.round(y)

    return TwoLayerPerceptron()

def zero_embedding(rows, columns):
    e = nn.Embedding(rows, columns)
    e.weight.data.zero_()
    return e

class factorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding( n_users,n_factors)
        self.item_factors = nn.Embedding( n_items,n_factors)
        self.user_bias = zero_embedding( n_users, 1)
        self.item_bias = zero_embedding( n_items, 1)
        self.fc = nn.Linear(n_factors, 4)
        
    def forward(self, X):
        user = X[:,0] - 1
        item = X[:,1] - 1
        return (self.user_factors(user) * self.item_factors(item)).sum(1) + self.user_bias(user).squeeze() + self.item_bias(item).squeeze()
