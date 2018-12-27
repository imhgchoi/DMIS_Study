import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from news_attention import NewsAttention

class HAN(nn.Module):
    def __init__(self, L, N):
        super(HAN, self).__init__()

        self.news_level_attention = [NewsAttention(L) for _ in range(N)]

    def forward(self, data):
        days = list()
        for i, network in enumerate(self.news_level_attention) :
            days.append(network.forward(data[i]))
        days = T.Tensor(days)









