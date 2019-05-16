import argparse
from state import State
from data_loader import load_data
from agent import Agent
from attention import Attention
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from evaluate import computeF1, evaluate
import math
import random

class ReactiveBaseline():
    def __init__(self, l):
        self.l = l
        self.b = 0.0
    def get_baseline_value(self):
        return self.b
    def update(self, target):
        self.b = (1-self.l)*self.b + self.l*target


GAMMA = 1
WORD_EMB_DIM = 4
NODE_EMB_DIM = 30
H_DIM = 16
T = 3
NUM_EPOCH = 100
SOFT_REWARD_SCALE = 0.1
NUM_ROLL_OUT = 5
SHUFFLE = True

# device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arguemnt parsing
parser = argparse.ArgumentParser("main.py")
parser.add_argument("dataset", help="the name of the dataset", type=str)
args = parser.parse_args()

# load dataset
rel_embedding, kg, train, test = load_data(args.dataset, WORD_EMB_DIM)

# projection from word embedding to node node embedding
word2node = nn.Linear(WORD_EMB_DIM, NODE_EMB_DIM, bias=False).to(device)

# mutihead self-attention
attention = Attention(1, NODE_EMB_DIM, H_DIM, math.sqrt(H_DIM)).to(device)

# list contains all params that need to optimize
model_param_list = list(word2node.parameters()) + list(attention.parameters())

# init agent
state = State((train[0][1],train[0][2]), kg, WORD_EMB_DIM, word2node, attention, rel_embedding, T, device) # init here to calculate the input size
input_dim = state.get_input_size()
num_rel = len(kg.rel_vocab)
num_entity = len(kg.en_vocab)
num_subgraph = len(state.subgraphs)
emb_dim = WORD_EMB_DIM + NODE_EMB_DIM
baseline = ReactiveBaseline(l=0.05)
agent = Agent(input_dim, 32, emb_dim, 0, 2, num_entity, num_rel,num_subgraph, GAMMA, 0.0001, model_param_list, baseline, device)

# training loop
index_list = list(range(len(train)))
for epoch in range(NUM_EPOCH):
    losses = []
    rewards = []
    correct = 0
    f1 = []
    if SHUFFLE:
        random.shuffle(index_list)
    for n in tqdm(range(len(train))):
        # create state from the question
        i = index_list[n]
        for _ in range(NUM_ROLL_OUT):
            state = State((train[i][1],train[i][2]), kg, WORD_EMB_DIM, word2node, attention, rel_embedding, T, device)
            answer = kg.en_vocab[train[i][0]]
            e0 = state.subgraphs[0][0]
            agent.policy.init_path(e0, state)

            # go for T step
            for step in range(T):
                action = agent.get_action(state)
                g, r, e = action
                if step < T-1:
                    agent.hard_reward(0)
                else:
                    nodes = state.get_last_nodes()
                    max_shortest_path = kg.max_shortest_path(nodes)
                    if answer == e and max_shortest_path == 0:
                        print("yes")
                        agent.hard_reward(10)
                    else:
                        answer_embedding = state.node_embedding[answer]
                        e_embedding = state.node_embedding[e]
                        agent.soft_reward(answer_embedding, e_embedding, SOFT_REWARD_SCALE, -max_shortest_path)
                state.update(action)
                #print("step: " + str(step) + ", take action: " + str(action) + "result_subgraphs:" + str(state.subgraphs))

            # compute f1
            f1.append(computeF1(answer, e)[-1])
            # update the policy net and record loss
            loss, reward, last_reward = agent.update_policy()
            if last_reward == 1:
                correct += 1
            losses.append(loss)
            rewards.append(reward)

    acc = correct/(NUM_ROLL_OUT*len(train))
    avg_loss = np.mean(losses)
    avg_reward = np.mean(rewards)
    avg_f1 = np.mean(f1)
    print("epoch: {}, loss: {}, reward: {}, acc: {}, f1: {}".format(epoch, avg_loss, avg_reward, acc, avg_f1))

    # evaluate on test set
    if (epoch+1)%5 == 0:
        evaluate(test, agent, kg, T, WORD_EMB_DIM, word2node, attention, rel_embedding, device, 15)



