import json

ER = {}       # adjacency matrix   
KG = {}       # each row 
QUESTION = {}
inv_ER = {}
questions = {}
rel_vocab = None

def inverse_r(r):
    words = r.split("_")
    if "inverse" in words:
        return r[:-8]
    r = r + "_inverse"
    if r in rel_vocab:
        return r

with open("vocab/relation_vocab.json") as json_file:
    rel_vocab = json.load(json_file)

with open("wc_kg.txt") as f:
    for row in f:
        row = row[:-1]
        elements = row.split('\t')
        relation = (elements[1], elements[2])
        entity = elements[0]
        invr = (elements[1], elements[0])
        inve = elements[2] 
        if entity not in ER:
            ER[entity] = [relation]
        else:
            ER[entity].append(relation)
        if inve not in inv_ER:
            inv_ER[inve] = [invr]
        else:
            inv_ER[inve].append(invr)
        KG[row] = 1

with open("wc_test_old.txt") as f:
    for row in f:
        row = row[:-1]
        a, r1, e1, r2, e2 = row.split('\t')
        r1 = inverse_r(r1)
        r2 = inverse_r(r2)
        e1_r1 = []
        for r, e in inv_ER[e1]:
            if r == r1:
                e1_r1.append(e)
        answers = []
        for r, e in inv_ER[e2]:
            if r == r2 and e in e1_r1:
                answers.append(e)
        question = r1 + '\t' + e1 + '\t' + r2 + '\t'+ e2
        str_answers = ";"
        for a in answers:
            str_answers += '\t' + a 
        if question not in questions:
            if question not in questions:
                questions[question] = 1
                out_line = question + str_answers
                print(out_line)

def inverse_r(r):
    words = r.split("_")
    if "inverse" in words:
        return r[:-8]
    r = r + "_inverse"
    if r in rel_vocab:
        return r

    

    
    
    
