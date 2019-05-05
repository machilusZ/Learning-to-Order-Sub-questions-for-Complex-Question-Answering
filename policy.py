import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim ,dropout_rate, lstm_num_layers, num_entity, num_rel):
        super(Policy, self).__init__()
        self.num_entity = num_entity
        self.action_dim = num_entity * (num_rel - 1) # we are not using start relation
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = 1             # currently using 1 question per batch

        # layers
        self.fc1 = nn.Linear(input_dim + hidden_dim, self.action_dim, bias=False)
        self.fc2 = nn.Linear(self.action_dim, self.action_dim, bias=False)
        self.Dropout1 = nn.Dropout(dropout_rate)
        #self.Dropout2 = nn.Dropout(dropout_rate)
        self.lstm_cell = nn.LSTM(input_size=self.action_dim,
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
        X = torch.cat((Ht,pt,Rt), dim=-1)

        X = self.fc1(X)
        X = F.relu(X)
        X = self.Dropout1(X)
        X = self.fc2(X)

        return X

    # init the first cell of LSTM
    def init_path(self, e0):
        # initial value for action with one hot encoding of (r0,e0)
        # where r0 is the DUMMY_START_RELATION (encoded as 0) and e0 is one entity from the question
        init_action = self.one_hot_encode((0,e0))
        init_action = init_action.view(self.batch_size, 1, -1)

        hidden_a = torch.randn(self.lstm_num_layers, self.batch_size, self.hidden_dim)
        hidden_b = torch.randn(self.lstm_num_layers, self.batch_size, self.hidden_dim)
        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)
        self.path = [self.lstm_cell(init_action, (hidden_a, hidden_b))[1]]

    # update path(history) by given an action format (r, e)
    def update_path(self, action):
        one_hot_action = self.one_hot_encode(action)
        one_hot_action = one_hot_action.view(self.batch_size, 1, -1)
        self.path.append(self.lstm_cell(one_hot_action, self.path[-1])[1])

    # one hot encode an action
    def one_hot_encode(self, action):
        r, e = action
        one_hot_action = torch.zeros(self.action_dim)
        index = r * self.num_entity + e
        one_hot_action[int(index)] = 1
        return one_hot_action




