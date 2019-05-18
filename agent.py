import torch
import numpy as np
import math
from policy import Policy
from scipy.spatial.distance import cosine
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F


class Agent():
    def __init__(self, input_dim, hidden_dim, emb_dim, dropout_rate, lstm_num_layers, num_entity, num_rel, num_subgraph, gamma, learning_rate, model_param_list, baseline, device):
        self.gamma = gamma
        self.num_subgraph = num_subgraph
        self.action_dim = emb_dim + num_subgraph
        self.num_entity = num_entity
        self.policy = Policy(input_dim, hidden_dim, emb_dim, dropout_rate, lstm_num_layers, num_entity, num_rel, num_subgraph, device).to(device)
        self.num_rel = num_rel
        self.reward_history = []
        self.logprob_history = []
        self.device = device
        params = list(self.policy.parameters()) + model_param_list
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        self.baseline = baseline

    def get_action(self, state):
        embedded_state = state.get_embedded_state()
        possible_actions = state.generate_all_possible_actions()
        out = self.policy(embedded_state)
        
        # build At (tsacked embeding of all possible actions)
        At = []
        for action in possible_actions:
            At.append(self.policy.encode(action, state))
        
        At = torch.stack(At).to(self.device)
        scores = torch.mv(At, out)
        scores = F.softmax(scores, dim=-1)

        # sample an action from the distribution
        c =  Categorical(scores)
        index = c.sample()
        action = possible_actions[index]
        g, r, e = action

        # add log prob to history
        self.logprob_history.append(c.log_prob(index))

        # add action to path
        self.policy.update_path((g,r,e), state)

        return (g,r,e)
    
    def get_probs(self, state, possible_actions):
        embedded_state = state.get_embedded_state()
        possible_actions = state.generate_all_possible_actions()
        out = self.policy(embedded_state)
        
        # build At (tsacked embeding of all possible actions)
        At = []
        for action in possible_actions:
            At.append(self.policy.encode(action, state))
        
        At = torch.stack(At).to(self.device)
        scores = torch.mv(At, out)
        scores = F.softmax(scores, dim=-1)

        # sample an action from the distribution
        c =  Categorical(scores)
        probs = np.zeros(len(possible_actions))
        for i, action in enumerate(possible_actions):
            probs[i] = math.exp(c.log_prob(torch.tensor([i])))
        return probs

    # assign hard reward
    def hard_reward(self, a):
        self.reward_history.append(a)

    # soft reward for the final step
    def soft_reward(self, answer_embedding, last_action_embedding, scale, path_reward):
        R = 1 - cosine(answer_embedding, last_action_embedding) + path_reward
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
        final_rewards = rewards - self.baseline.get_baseline_value() 

        # Calculate loss
        logprobs = torch.stack(self.logprob_history)
        loss = (torch.sum(torch.mul(logprobs, Variable(final_rewards).to(self.device)).mul(-1), -1))
        self.baseline.update(torch.mean(rewards))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # reinitialize history
        self.clear_history()

        hit = 0
        for i in rewards:
            if i == 1:
                hit += 1

        return loss.item(), torch.sum(rewards).item(), hit

    def clear_history(self):
        # reinitialize history
        self.logprob_history = []
        self.reward_history = []
