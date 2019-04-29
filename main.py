import argparse
from state import State
from data_loader import load_data
from agent import Agent

GAMMA = 0.5
EMB_DIM = 300
T = 2

parser = argparse.ArgumentParser("main.py")
parser.add_argument("dataset", help="the name of the dataset", type=str)
args = parser.parse_args()

# load dataset
kg, train, test = load_data(args.dataset)

# test: train on the first question
# create state from the question
state = State((train[0][1],train[0][2]), kg, 300)
answer = kg.en_vocab[train[0][0]]

# init agent
input_dim = state.get_input_size()
num_rel = len(kg.rel_vocab)
num_entity = len(kg.en_vocab)
agent = Agent(input_dim, 5, 0.5, 1, num_entity, num_rel, 0.9, 0.0001)

e0 = state.subgraphs[0][0]
agent.policy.init_path(e0)

for i in range(T):
    embedded_state = state.get_embedded_state()
    possible_actions = state.generate_all_possible_actions()
    action = agent.get_action(embedded_state, possible_actions)
    r, e = action
    if i < T-1:
        agent.hard_reward(0)
    else:
        if answer == e:
            agent.hard_reward(1)
        else:
            answer_embedding = state.node_embedding[answer]
            e_embedding = state.node_embedding[e]
            agent.soft_reward(answer_embedding, e_embedding)
    state.update(action)
    print("step: " + str(i) + ", take action: " + str(action) + "result_subgraphs:" + str(state.subgraphs))
agent.update_policy()



