import torch
import numpy as np
import math
from policy import Policy 
from scipy.spatial.distance import cosine
from torch.distributions import Categorical
from torch.autograd import Variable


class Agent():
    def __init__(self, input_dim, hidden_dim ,dropout_rate, lstm_num_layers, num_entity, num_rel, gamma, learning_rate, model_param_list):
        self.gamma = gamma
        self.action_dim = num_entity * num_rel
        self.num_entity = num_entity
        self.policy = Policy(input_dim, hidden_dim ,dropout_rate, lstm_num_layers, num_entity, num_rel)
        self.reward_history = []
        self.logprob_history = []
        params = list(self.policy.parameters()) + model_param_list
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def get_action(self, state, possible_actions):
        scores = self.policy(state)
        # zero out all impossible actions
        possible_index = []
        for action in possible_actions:
            r, e = action
            index = r * self.num_entity + e
            possible_index.append(index)
        scores_possible = scores[possible_index]
        sm = torch.nn.Softmax(dim=-1)
        scores_possible = sm(scores_possible)

        # sample an action from the distribution
        c =  Categorical(scores_possible)
        index = c.sample()
        action = possible_index[index]
        r = math.floor(action/self.num_entity)
        e = action%self.num_entity

        # add log prob to history
        self.logprob_history.append(c.log_prob(index))

        # add action to path
        self.policy.update_path((r,e))

        return (r,e)

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

        return loss.item(), torch.sum(rewards).item(), rewards[-1]
