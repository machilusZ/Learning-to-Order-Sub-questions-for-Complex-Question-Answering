"""
Some helper scripts to quantitatively compute the performance
of the model
"""
import numpy as np
from tqdm import tqdm
import torch
from state import State
import math
import copy

def computeF1(goldList, predictedList):
    '''
    return a tuple with recall, precision, and f1 for one example. This
    scripted was originally implemented with the assumption that the questions
    might have multiple solutions. With small fix, this could be adapt to
    single-solution case as well.
    Adapted from https://github.com/ysu1989/GraphQuestions/blob/255ebc92301f93afd5b98165e431833a3cba38e7/evaluate.py
    '''

    # convert the prediction and ground truth label to list
    # in the case of single-solution question
    if not isinstance(goldList, list):
        goldList = [goldList]
    if not isinstance(predictedList, list):
        predictedList = [predictedList]

    # Assume questions have at least one answer
    if len(goldList) == 0:
        raise Exception('gold list may not be empty')
    # If we return an empty list recall is zero and precision is one
    if len(predictedList) == 0:
        return (0, 1, 0)
    # It is guaranteed now that neither of the lists is empty

    precision = 0
    for entity in predictedList:
        if entity in goldList:
            precision += 1
    precision = float(precision) / len(predictedList)

    recall = 0
    for entity in goldList:
        if entity in predictedList:
            recall += 1
    recall = float(recall) / len(goldList)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)
    return (recall, precision, f1)

def evaluate(test, agent, kg, T, WORD_EMB_DIM, word2node, attention, rel_embedding, device, beam_size):
    """
    A function to run evaluation on the test set
    """
    hit_1 = 0
    hit_2 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    with torch.no_grad():
        f1 = []
        for i in tqdm(range(len(test))):
            # for each test question
            # create beam size # of state and agents
            states = []
            for _ in range(beam_size):
                state = State((test[i][1],test[i][2]), kg, WORD_EMB_DIM, word2node, attention, rel_embedding, T, device)
                states.append(state)
            
            answer = kg.en_vocab[test[i][0]]
            e0 = state.subgraphs[0][0]
            agent.policy.init_path(e0, state)

            # the first step
            embedded_state = states[0].get_embedded_state()
            possible_actions = state.generate_all_possible_actions()
            probs, index = agent.get_probs(embedded_state, possible_actions)
            top_index = np.argsort(probs)[::-1][0:beam_size]
            top_path_probs = probs[top_index]
            top_index = index[top_index]
            top_path = []

            for index, i in enumerate(top_index):
                g = math.floor(i/(agent.num_entity*agent.num_rel))
                rest = i%(agent.num_entity*agent.num_rel)
                r = math.floor(rest/agent.num_entity)
                e = i%agent.num_entity
                top_path.append([(g,r,e)])
                states[index].update((g,r,e))
        
            # go for T - 1 step
            for step in range(1, T):
                top_path, top_path_probs, states = get_tops(agent, top_path, top_path_probs, states, e0, beam_size)
            
            final_entities = []
            for path in top_path:
                temp = path[-1][2]
                final_entities.append(temp)
            
            final_entities += [-1]*10
            ranked_1  = top_path[0][-1][-1]
            ranked_10 = final_entities[0:10]
            ranked_2  = final_entities[0:2]
            ranked_3  = final_entities[0:3]
            ranked_5  = final_entities[0:5]

            if answer == ranked_1:
                hit_1 += 1
            if answer in ranked_10:
                hit_10 += 1
            if answer in ranked_5:
                hit_5 += 1
            if answer in ranked_3:
                hit_3 += 1
            if answer in ranked_2:
                hit_2 += 1
        
        hit_1 /= len(test)
        hit_10/= len(test)
        hit_5/= len(test)
        hit_3/= len(test)
        hit_2/= len(test)
        print("hit@1: " + str(hit_1) + ", hit@2: " + str(hit_2)+ ", hit@3: " + str(hit_3) + ", hit@5: " + str(hit_5) + ", hit@10: " + str(hit_10))
        # compute f1
        '''    
            f1.append(computeF1(answer, e)[-1])
        avg_f1 = np.mean(f1)
        print("Average test f1: {}".format(avg_f1))
        agent.clear_history()
        '''

def get_tops(agent, top_path, top_path_probs, states, e0, beam_size):
    prob2action = {}
    all_probs = []
    next_path_probs = []
    next_top_path = []
    new_states = []
    for index, actions in enumerate(top_path):
        agent.policy.init_path(e0, states[index])
        for action in actions:
            agent.policy.update_path(action, states[index])
        embedded_state = states[index].get_embedded_state()
        possible_actions = states[index].generate_all_possible_actions()
        probs, possible_index = agent.get_probs(embedded_state, possible_actions)
        top_index = np.argsort(probs)[::-1][0:beam_size]
        top_probs = probs[top_index]
        top_index = possible_index[top_index]
        for i, value  in enumerate(top_index):
            g = math.floor(value/(agent.num_entity*agent.num_rel))
            rest = value%(agent.num_entity*agent.num_rel)
            r = math.floor(rest/agent.num_entity)
            e = value%agent.num_entity
            prob = top_probs[i] * top_path_probs[index]
            all_probs.append(prob)
            prob2action[prob] = (index,g,r,e)
    top_index = np.argsort(all_probs)[::-1][0:beam_size]
    for i in top_index:
        prob = all_probs[i]
        next_path_probs.append(prob)
        index, g, r, e = prob2action[prob]
        path = copy.deepcopy(top_path[index])
        path = path + [(g,r,e)]
        next_top_path.append(path)
        state = copy.deepcopy(states[index])
        state.update((g,r,e))
        new_states.append(state)
    return next_top_path, next_path_probs, new_states

        



