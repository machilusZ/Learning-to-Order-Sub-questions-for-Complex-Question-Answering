import argparse
from state import State
from data_loader import load_data
from agent import Agent
from attention import Attention
import torch.nn as nn
import numpy as np
from tqdm import tqdm

GAMMA = 0.5
WORD_EMB_DIM = 300
NODE_EMB_DIM = 16
H_DIM = 16
T = 3
NUM_EPOCH = 10

parser = argparse.ArgumentParser("main.py")
parser.add_argument("dataset", help="the name of the dataset", type=str)
args = parser.parse_args()

# load dataset
kg, train, test = load_data(args.dataset)

# projection from word embedding to node node embedding
word2node = nn.Linear(WORD_EMB_DIM, NODE_EMB_DIM, bias=False)      

# mutihead self-attention 
attention = Attention(2, NODE_EMB_DIM, H_DIM, 0.01)                

# list contains all params that need to optimize
model_param_list = list(word2node.parameters()) + list(attention.parameters())

# init agent
state = State((train[0][1],train[0][2]), kg, WORD_EMB_DIM, word2node, attention) # init here to calculate the input size
input_dim = state.get_input_size()
num_rel = len(kg.rel_vocab)
num_entity = len(kg.en_vocab)
agent = Agent(input_dim, 5, 0.5, 2, num_entity, num_rel, 0.1, 0.001, model_param_list)

# training loop
for epoch in range(NUM_EPOCH):
    losses = []
    for i in tqdm(range(len(train))):
        # create state from the question
        state = State((train[i][1],train[i][2]), kg, WORD_EMB_DIM, word2node, attention)
        answer = kg.en_vocab[train[i][0]]
        e0 = state.subgraphs[0][0]
        agent.policy.init_path(e0)
        
        # go for T step
        for step in range(T):
            embedded_state = state.get_embedded_state()
            possible_actions = state.generate_all_possible_actions()
            action = agent.get_action(embedded_state, possible_actions)
            r, e = action
            if step < T-1:
                agent.hard_reward(0)
            else:
                if answer == e:
                    agent.hard_reward(1)
                else:
                    answer_embedding = state.node_embedding[answer]
                    e_embedding = state.node_embedding[e]
                    agent.soft_reward(answer_embedding, e_embedding)
            state.update(action)
            #print("step: " + str(step) + ", take action: " + str(action) + "result_subgraphs:" + str(state.subgraphs))
        
        # update the policy net and record loss
        loss = agent.update_policy()
        losses.append(loss)

    avg_loss = np.mean(losses)
    print("epoch: " + str(epoch) + ", loss: " + str(loss))



