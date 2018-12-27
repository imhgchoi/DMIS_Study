import torch as T
import torch.nn as nn
import torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self, L):
        super(BiGRU, self).__init__()


        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        pass









