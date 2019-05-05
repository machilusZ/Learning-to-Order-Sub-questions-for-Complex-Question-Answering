import torch
import numpy as np
import math
from policy import Policy 
from scipy.spatial.distance import cosine
from torch.distributions import Categorical
from torch.autograd import Variable


class Agent():
    def __init__(self, input_dim, hidden_dim ,dropout_rate, lstm_num_layers, word_emb_dim, node_emb_dim, gamma, learning_rate, model_param_list):
        self.gamma = gamma
        self.action_dim = word_emb_dim + node_emb_dim
        self.policy = Policy(input_dim, hidden_dim ,dropout_rate, lstm_num_layers, word_emb_dim, node_emb_dim)
        self.reward_history = []
        self.logprob_history = []
        self.sm = torch.nn.Softmax(dim=-1)
        params = list(self.policy.parameters()) + model_param_list
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def get_action(self, state, possible_actions, possible_actions_emb):
        out = self.policy(state)
        At = torch.Tensor(possible_actions_emb)
        scores = torch.mv(At, out) 
        scores = self.sm(scores)

        # sample an action from the distribution
        c =  Categorical(scores)
        index = c.sample()
        action = possible_actions[index]
        action_emb = possible_actions_emb[index]

        # add log prob to history
        self.logprob_history.append(c.log_prob(index))

        # add action to path
        self.policy.update_path(action_emb)

        return action

    # assign hard reward
    def hard_reward(self, a):
        self.reward_history.append(a)

    # soft reward for the final step
    def soft_reward(self, answer_embedding, last_action_embedding, scale):
        R = 1 - cosine(answer_embedding, last_action_embedding)
        self.reward_history.append(scale*R)
           
    def update_policy(self):
        R = 0
        rewards = []
        for r in self.reward_history[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)
       
        # Scale rewards
        rewards = np.array(rewards)
        # if we normalize the reward, the loss don't go down
        # rewards = (rewards - rewards.mean()) / (np.std(rewards) + np.finfo(np.float32).eps)
        rewards = torch.Tensor(rewards)

        # Calculate loss
        logprobs = torch.stack(self.logprob_history)
        loss = (torch.sum(torch.mul(logprobs, Variable(rewards)).mul(-1), -1))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # reinitialize history
        self.logprob_history = []
        self.reward_history = []

        hit = 0
        for i in rewards:
            if i == 1:
                hit += 1

        return loss.item(), torch.sum(rewards).item(), hit
