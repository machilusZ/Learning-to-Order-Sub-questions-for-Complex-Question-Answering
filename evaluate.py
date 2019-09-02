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

def evaluate(test, agent, kg, T, WORD_EMB_DIM, word2node, attention, rel_embedding, node_embedding, device, beam_size):
    """
    A function to run evaluation on the test set
    """
    hit_1 = 0
    hit_2 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    with torch.no_grad():
        cumulated_risks = []
        picked_path_risks = []
        picked_path_count = 0
        for i in tqdm(range(len(test))):
            # for each test question
            # create beam size # of state and agents
            states = []
            #print(test[i])
            for _ in range(beam_size):
                state = State((test[i][1],test[i][2]), kg, node_embedding, WORD_EMB_DIM, word2node, attention, rel_embedding, T, device)
                states.append(state)
            
            '''
            for r in test[i][2]:
                encoded_r = kg.rel_vocab[r] - 2 # remove no-ops and dummy-start
                if encoded_r not in subquestions:
                    subquestions.append(encoded_r)
            '''

            answers = kg.encode_answers(test[i][0])
            e0 = state.subgraphs[0][0]
            agent.policy.init_path(e0, state)

            # get subquestions
            e1, e2 = test[i][1]
            r1, r2, r3 = test[i][2]
            e3 = test[i][3] 
            subquestions = [(kg.rel_vocab[r1], answers), (kg.rel_vocab[r2], [kg.en_vocab[e3]]), (kg.rel_vocab[r3], answers)]
            

            # the first step
            possible_actions = np.array(state.generate_all_possible_actions())
            probs = agent.get_probs(states[0], possible_actions)
            top_index = np.argsort(probs)[::-1][0:beam_size]
            top_path_probs = probs[top_index]

            risks_foreach_steps = []
            risks = calculate_risk(possible_actions, probs, subquestions)
            risks_foreach_steps.append(risks)

            top_actions = possible_actions[top_index]
            top_path = []

            for index, action in enumerate(top_actions):
                top_path.append([action])
                states[index].update(action)
        
            # go for T - 1 step
            for step in range(1, T):
                top_path, top_path_probs, states, risks = get_tops(agent, top_path, top_path_probs, states, e0, beam_size, subquestions)
                risks_foreach_steps.append(risks)

            if len(cumulated_risks) == 0:
                cumulated_risks = np.array(risks_foreach_steps)
            else:
                cumulated_risks += np.array(risks_foreach_steps)

            picked_path_risk = []
            r1 = kg.rel_vocab[r1]
            r2 = kg.rel_vocab[r2]
            r3 = kg.rel_vocab[r3]
            for step, action in enumerate(top_path[0]):
                g, r, e = action
                if r == r1:
                    r1 = -1 # already picked
                    picked_path_risk.append(risks_foreach_steps[step][0])
                elif r == r2:
                    r2 = -1 # already picked
                    picked_path_risk.append(risks_foreach_steps[step][1])
                elif r == r3:
                    r3 = -1 # already picked
                    picked_path_risk.append(risks_foreach_steps[step][2])
                else:
                    picked_path_risk = []
                    break
            
            if len(picked_path_risk) != 0:
                picked_path_count += 1 
                if len(picked_path_risks) == 0:
                    picked_path_risks = np.array(picked_path_risk)
                else:
                    picked_path_risks += np.array(picked_path_risk)

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

            if ranked_1 in answers:
                hit_1 += 1
            if correct(ranked_10, answers):
                hit_10 += 1
            if correct(ranked_5, answers):
                hit_5 += 1
            if correct(ranked_3, answers):
                hit_3 += 1
            if correct(ranked_2, answers):
                hit_2 += 1
        
        hit_1 /= len(test)
        hit_10/= len(test)
        hit_5/= len(test)
        hit_3/= len(test)
        hit_2/= len(test)      

        avg_risks = cumulated_risks/len(test)
        avg_path_risks = picked_path_risks/picked_path_count
        print("hit@1: " + str(hit_1) + ", hit@2: " + str(hit_2)+ ", hit@3: " + str(hit_3) + ", hit@5: " + str(hit_5) + ", hit@10: " + str(hit_10))
        print(avg_risks)
        print(avg_path_risks)

        '''    
            f1.append(computeF1(answer, e)[-1])
        avg_f1 = np.mean(f1)
        print("Average test f1: {}".format(avg_f1))
        agent.clear_history()
        '''

def get_tops(agent, top_path, top_path_probs, states, e0, beam_size, subquestions):
    prob2action = {}
    all_probs = []
    next_path_probs = []
    next_top_path = []
    new_states = []
    for index, actions in enumerate(top_path):
        agent.policy.init_path(e0, states[index])
        for action in actions:
            agent.policy.update_path(action, states[index])
        possible_actions = np.array(states[index].generate_all_possible_actions())
        probs = agent.get_probs(states[index], possible_actions)

        # calculate the Risk of each subquestion
        risks = calculate_risk(possible_actions, probs, subquestions)

        top_index = np.argsort(probs)[::-1][0:beam_size]
        top_probs = probs[top_index]
        top_actions = possible_actions[top_index]
        for i, value  in enumerate(top_index):
            g, r, e = possible_actions[value]
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
    return next_top_path, next_path_probs, new_states, risks

def correct(pred_list, answers):
    for a in answers:
        if a in pred_list:
            return True

    return False

def calculate_risk(possible_actions, probs, subquestions):
    # build dict for looking up the prob for a certain action and dict for looking up prob for a certain r
    act2prob = {}
    r2prob = {}
    for index, action in enumerate(possible_actions):
        key = generate_key(action[0], action[1], action[2])
        act2prob[key] = probs[index]
        
        r = action[1]
        if r not in r2prob:
            r2prob[r] = 0
        r2prob[r] += probs[index]
    
    # calculate risk for each subquestions
    calculated_risks = []
    for r, answers in subquestions:
        P_aq = 0
        for a in answers:
            # look up in the first subgraph
            search_key = generate_key(0, r, a)
            if search_key in act2prob:
                P_aq += act2prob[search_key]
        
            # look up in the second subgraph
            search_key = generate_key(1, r, a)
            if search_key in act2prob:
                P_aq += act2prob[search_key]

        risk = 1
        if r in r2prob:
            P_q = r2prob[r]
            risk = (P_q - P_aq)/P_q

        calculated_risks.append(risk)

    return calculated_risks

def generate_key(g, r, e):
    return "" + str(g) + ";" + str(r) + ";" + str(e)



