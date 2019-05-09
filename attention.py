import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
solution for the problem: nn.Module not importing parameters contained in lists
Adapted from https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219
'''
class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class Attention(nn.Module):
    def __init__(self, num_of_head, input_dim, output_dim, scale_factor):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_of_head = num_of_head
        self.scale_factor = scale_factor
        self.Wqs = ListModule(self, 'wqs')
        self.Wks = ListModule(self, 'wks')
        self.Wvs = ListModule(self, 'wvs')
        # TODO: check dim

        for _ in range(num_of_head):

            Wq = nn.Linear(input_dim,output_dim)
            Wk = nn.Linear(input_dim,output_dim)
            Wv = nn.Linear(input_dim,output_dim)

            self.Wqs.append(Wq)
            self.Wks.append(Wk)
            self.Wvs.append(Wv)

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
            score = F.softmax(score, dim=0)
            z = torch.mm(score, v)
            Z.append(z)
        Z = torch.cat(Z, dim=1)
        return torch.sigmoid(Z)

        
