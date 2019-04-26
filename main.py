import argparse
from state import State
from data_loader import load_data

GAMMA = 0.5
EMB_DIM = 300

parser = argparse.ArgumentParser("main.py")
parser.add_argument("dataset", help="the name of the dataset", type=str)
args = parser.parse_args()

# load dataset
kg, train, test = load_data(args.dataset)

# create state from the question
state = State((train[0][1],train[0][2]), kg, GAMMA, EMB_DIM)


