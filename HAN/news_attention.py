import torch as T
import torch.nn as nn
import torch.nn.functional as F

class NewsAttention(nn.Module):
    def __init__(self, L):
        super(NewsAttention, self).__init__()
        self.attention = nn.Linear(50,1, bias=True)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        alpha = F.softmax(F.sigmoid(self.attention(data)))
        return self.attention(alpha)









