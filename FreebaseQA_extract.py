import argparse
import json
import random
parser = argparse.ArgumentParser()
parser.add_argument(
    '-input',
    dest='input',
    help='input file',
    type=str
)

parser.add_argument(
    '-hops',
    dest='hops',
    help='number of hops',
    type=str
)

parser.add_argument(
    '-num_samples',
    dest='num_samples',
    help='number of samples',
    type=str
)

parser.add_argument(
    '-out',
    dest='out_file',
    help='output file',
    type=str
)

e2r = {}
er2a = {}
args = parser.parse_args()

# parse json
with open(args.input, 'r') as f:
    questions = json.load(f)['Questions']
    for question in questions:
        for parse in question['Parses']:
            e = (parse['TopicEntityName'], parse['TopicEntityMid'])
            if e not in e2r:
                e2r[e] = set()

            infer_chain = parse['InferentialChain']
            r_list = infer_chain.split('..') # I assume .. means a seperate path in the chain
            a_list = parse['Answers']

            for r in r_list:
                if r not in e2r[e]:
                    e2r[e].add(r)
                if (e,r) not in er2a:
                    er2a[(e,r)] = set()
                for a in a_list:
                    for a_name in a['AnswersName']: # it seems a answer can have multiple names
                        if (a_name, a['AnswersMid']) not in er2a[(e,r)]:
                            er2a[(e,r)].add((a_name, a['AnswersMid']))

total_fail = 0            
ret = set()
while len(ret) < int(args.num_samples) and total_fail < int(args.num_samples)*10:
    one_path = []
    e_on_path = set()

    e_idx = random.randint(0,len(e2r)-1)
    next_e = list(e2r.keys())[e_idx]
    one_fail = 0

    while len(one_path) < int(args.hops) and one_fail < 10:
        if next_e not in e2r: # dead path
            break

        r_idx = random.randint(0,len(e2r[next_e])-1)
        r = list(e2r[next_e])[r_idx]

        if next_e not in e_on_path:
            e_on_path.add(next_e)
            one_path.append((next_e,r))
            a_idx = random.randint(0,len(er2a[(next_e,r)])-1)
            next_e = list(er2a[(next_e,r)])[a_idx]
        else:   # need to re-choose relation
            one_fail += 1

    if len(one_path) >= int(args.hops):
        one_path = tuple(one_path)
        if (one_path, next_e) not in ret:
            ret.add((one_path, next_e))  # in the form of (path, answer)
        else:
            total_fail += 1
    else:
        total_fail += 1

"""
output format:
( ((e1_name,e1_id),r1), ((e2_name,e2_id),r2), ...,  ((en_name,en_id),rn) ), (a_name, a_id)  )
"""
with open(args.out_file, 'w') as f:
    for line in ret:
        f.write(str(line))
        f.write('\n')

