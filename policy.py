import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, emb_dim, dropout_rate, lstm_num_layers, num_entity, num_rel, num_subgraph, device):
        super(Policy, self).__init__()
        self.num_entity = num_entity
        self.action_dim = num_subgraph * num_entity * num_rel
        self.num_subgraph = num_subgraph
        self.num_rel = num_rel
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        self.batch_size = 1             # currently using 1 question per batch

        # layers
        self.fc1 = nn.Linear(input_dim + hidden_dim, input_dim + hidden_dim, bias=False)
        self.fc2 = nn.Linear(input_dim + hidden_dim, self.action_dim, bias=False)
        #self.Dropout1 = nn.Dropout(dropout_rate)
        #self.Dropout2 = nn.Dropout(dropout_rate)
        self.lstm_cell = nn.LSTM(input_size=emb_dim + num_subgraph,
                            hidden_size=self.hidden_dim,
                            num_layers=self.lstm_num_layers,
                            batch_first=True)
        # the array of ht
        self.path = None  

    def forward(self, state):
        Ht, Rt = state
        pt = self.path[-1][0][-1,:,:]
        
        # cancatenate
        Ht = torch.stack(Ht).view(-1)
        Rt = Rt.view(-1)
        pt = pt.view(-1)
        X = torch.cat((Ht,pt,Rt.to(self.device)), dim=-1)

        X = self.fc1(X)
        X = F.relu(X)
        #X = self.Dropout1(X)
        X = self.fc2(X)

        return X

    # init the first cell of LSTM
    def init_path(self, e0, state):
        # initial value for action with one hot encoding of (r0,e0)
        # where r0 is the DUMMY_START_RELATION (encoded as 0) and e0 is one entity from the question
        init_action = self.encode((0,0,e0), state)
        init_action = init_action.view(self.batch_size, 1, -1)

        hidden_a = torch.randn(self.lstm_num_layers, self.batch_size, self.hidden_dim)
        hidden_b = torch.randn(self.lstm_num_layers, self.batch_size, self.hidden_dim)
        hidden_a = Variable(hidden_a).to(self.device)
        hidden_b = Variable(hidden_b).to(self.device)
        self.path = [self.lstm_cell(init_action.to(self.device), (hidden_a, hidden_b))[1]]

    # update path(history) by given an action format (r, e)
    def update_path(self, action, state):
        one_hot_action = self.encode(action, state)
        one_hot_action = one_hot_action.view(self.batch_size, 1, -1)
        self.path.append(self.lstm_cell(one_hot_action.to(self.device), self.path[-1])[1])

    # one hot encode an action
    def encode(self, action, state):
        g, r, e = action
        one_hot_g = torch.zeros(self.num_subgraph)
        one_hot_g[int(g)] = 1
        rel_emb = torch.Tensor(state.rel_embedding[r])
        en_emb = torch.Tensor(state.node_embedding[e])
        return torch.cat((one_hot_g,rel_emb,en_emb), dim=-1)




