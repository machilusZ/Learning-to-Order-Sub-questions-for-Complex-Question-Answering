from knowledge_graph import KnowledgeGraph
from tqdm import tqdm
import numpy as np
import re

def load_data(dataset_name, word_emb_size, node_embed_type):
    graph_path = "data/" + dataset_name + "/" + dataset_name + "_kg.txt"
    train_path = "data/" + dataset_name + "/" + dataset_name + "_train.txt"
    test_path = "data/" + dataset_name + "/" + dataset_name +  "_test.txt"
    vocab_path = "data/" + dataset_name + "/vocab/"
    node_embed_path = "data/" + dataset_name + "/" + dataset_name +  "_embed_" + node_embed_type + ".npy"
    
    graph = KnowledgeGraph(graph_path, vocab_path)
    test = load_questions(train_path, parser)
    train = load_questions(test_path, parser)
    rel_embedding = init_rel_embedding("glove.840B.300d.txt", camel_case_spliter, word_emb_size, graph)
    node_embedding = np.load(node_embed_path)
    
    return node_embedding, rel_embedding, graph, test, train


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
    return (answers, [e1,e2], [r1,r2,r3], internal_node)

def wc_parser(line):
    temp = line.decode("utf-8").strip().split(";")
    question = temp[0]
    answer = temp[1]
    r1, e1, r2, e2 = question.split("\t")
    answers = answer.split("\t")[1:]
    return (answers, [e1,e2], [r1,r2])

def init_rel_embedding(path_to_embedding, spliter, word_emb_size, graph):
    # read in the embeding
    embeddings_index = {}
    rel_embedding = {}
    
    '''
    with open(path_to_embedding) as f:
        for line in tqdm(f):
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                pass
    '''
               
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

        # if all words of a relation are not in our pretrained glove, set to one hot
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



