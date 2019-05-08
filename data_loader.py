from knowledge_graph import KnowledgeGraph
from tqdm import tqdm
import numpy as np
import re

def load_data(dataset_name, word_emb_size):
    graph_path = "data/" + dataset_name + "/" + dataset_name + "_kg.txt"
    train_path = "data/" + dataset_name + "/" + dataset_name + "_train.txt"
    test_path = "data/" + dataset_name + "/" + dataset_name +  "_test.txt"
    vocab_path = "data/" + dataset_name + "/vocab/"
    graph = KnowledgeGraph(graph_path, vocab_path)
    test = load_questions(train_path, countires_parser)
    train = load_questions(test_path, countires_parser)
    rel_embedding = init_rel_embedding("glove.840B.300d.txt", camel_case_spliter, word_emb_size, graph)
    return rel_embedding, graph, test, train


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
    temp = line.decode("utf-8").strip().split("\t")
    if len(temp) != 5:
        return None
    a, r1, e1, r2, e2 = temp
    return (a, [e1,e2], [r1,r2])

def init_rel_embedding(path_to_embedding, spliter, word_emb_size, graph):
    # read in the embeding
    embeddings_index = {}
    rel_embedding = {}
    with open(path_to_embedding) as f:
        for line in tqdm(f):
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                pass
                
    for r in graph.rel_vocab:
        index = graph.rel_vocab[r]
        words = spliter(r)
        r_vector = np.zeros((word_emb_size))
        found = 0
        for word in words:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is not None:
                found += 1
                r_vector += embedding_vector
        
        # if all words of a relation are not in our pretrained glove, set to ran
        if found == 0:
            rel_embedding[index] = np.zeros((word_emb_size))
            rel_embedding[index][index] = 1
        else:
            rel_embedding[index] = r_vector/found
    return rel_embedding

# spliters: split the relation into words
def camel_case_spliter(word):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',word)
    return [m.group(0) for m in matches]



