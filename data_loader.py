from knowledge_graph import KnowledgeGraph

def load_data(dataset_name):
    graph_path = "data/" + dataset_name + "/" + dataset_name + "_kg.txt"
    train_path = "data/" + dataset_name + "/" + dataset_name + "_train.txt"
    test_path = "data/" + dataset_name + "/" + dataset_name +  "_test.txt"
    vocab_path = "data/" + dataset_name + "/vocab/"
    graph = KnowledgeGraph(graph_path, vocab_path)
    test = load_questions(train_path, countires_parser)
    train = load_questions(test_path, countires_parser)
    return graph, test, train


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
def countires_parser(line):
    temp = line[0:-2].decode("utf-8").split("\t")
    if len(temp) != 5:
        return None
    a, r1, e1, r2, e2 = temp
    return (a, [e1,e2], [r1,r2])



