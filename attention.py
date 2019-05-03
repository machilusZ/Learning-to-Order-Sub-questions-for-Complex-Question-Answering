import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, num_of_head, input_dim, output_dim, scale_factor):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_of_head = num_of_head
        self.scale_factor = scale_factor
        self.Wqs = []
        self.Wks = []
        self.Wvs = []
        # TODO: check dim
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

        for _ in range(num_of_head):
            self.Wqs.append(nn.Linear(input_dim,output_dim))
            self.Wks.append(nn.Linear(input_dim,output_dim))
            self.Wvs.append(nn.Linear(input_dim,output_dim))

    def forward(self, X):
        Z = []
        for i in range(self.num_of_head):
            Wq = self.Wqs[i]
            Wk = self.Wks[i]
            Wv = self.Wvs[i]
            q = Wq(X)
            k = Wk(X)
            v = Wv(X)
            score = self.scale_factor * torch.mm(q,torch.t(k))
            score = self.softmax(score)
            z = torch.mm(score, v)
            Z.append(z)
        Z = torch.cat(Z, dim=1)
        return self.sigmoid(Z)

        
