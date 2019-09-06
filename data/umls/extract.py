def load_questions(file_path, line_parser):
    with open(file_path, 'rb') as fp:
        line = fp.readline()
        qs = []
        while line:
            question = line_parser(line)
            if question:
                qs.append(question)
            line = fp.readline()
    return qs

# Parsers, each parser takes in one line of the question file and return (answer, [e1, e2, ..], [r1, r2, ...])
def parser(line):
    temp = line.decode("utf-8").strip().split(";")
    question = temp[0]
    answer = temp[1]
    internal_node = temp[2]
    r1, e1, r2, r3, e2 = question.split("\t")
    answers = answer.split("\t")[1:]
    return (answers, e1, e2, r1,r2,r3, internal_node.split("\t")[1:])

train_path = "umls_train.txt"
test_path = "umls_test.txt"
test = load_questions(test_path, parser)
train = load_questions(train_path, parser)

with open("train.txt", 'w') as f1:
    for question in train:
        answers, e1, e2, r1,r2,r3, internal_node = question
        f1.write(e2 + "\t" + r2 + "\t" + internal_node[0] + "\n")
    
with open("test.txt", 'w') as f2:
    for question in test:
        answers, e1, e2, r1,r2,r3, internal_node = question
        f2.write(e2 + "\t" + r2 + "\t" + internal_node[0] + "\n")


