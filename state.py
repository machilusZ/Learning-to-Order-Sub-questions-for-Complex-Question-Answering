import numpy as np
from tqdm import tqdm
import re

class State:
    def __init__(self, question, graph, gamma, word_emb_dim):
        print(question)
        self.es, self.rs = graph.encode_question(question) # encoded entities and relations in the question
        self.word_emb_dim = word_emb_dim  # dimension of word embeding
        self.subgraphs = []     # each element is a vector of format [e1, e2, ... ] representing a subgraph
        self.rel_embedding = {} # mapping from all the relations in vocb to its embeding
        self.Rt = []            # Rt in the state
        self.Ht  = None          # TODO: node embedings
        self.gamma = gamma
        self.graph = graph      # knowledge graph object

        # init all subgraphs from the question
        for e in self.es:
            self.subgraphs.append([e])

        # get word vectors for each relations
        self.init_embedding("glove.840B.300d.txt",camel_case_spliter)

        # init Rt from the relations in the question
        self.init_Rt()

    # add a node to one of the subgraph, input format (e1, r, e2)
    def update(self, edge):
        e1, _, e2 = edge
        for subgraph in self.subgraphs:
            if e1 in subgraph:
                subgraph.append(e2)

    # find the neighbors of all subgraphs
    def find_all_neighbors(self):
        neighbors = []
        for subgraph in self.subgraphs:
            ret = self.find_subgraph_neighbors(subgraph)
            neighbors += ret
        return neighbors

    # helper function: find the neighbor edge (e1, r, e2) of a subgraph
    def find_subgraph_neighbors(self, subgraph):
        ret = []
        for e1 in subgraph:
            neighbors = self.graph.graph[e1]
            for (r, e2) in neighbors:
                if e2 not in subgraph:
                    ret.append((e1, r, e2))
        return ret

    # embed relations
    def init_embedding(self, path_to_embedding, spliter):
        # read in the embeding
        embeddings_index = {}

        # with open(path_to_embedding) as f:
        #     for line in tqdm(f):
        #         try:
        #             values = line.split()
        #             word = values[0]
        #             coefs = np.asarray(values[1:], dtype='float32')
        #             embeddings_index[word] = coefs
        #         except:
        #             pass

        for r in self.graph.rel_vocab:
            index = self.graph.rel_vocab[r]
            words = spliter(r)
            r_vector = np.zeros((self.word_emb_dim))
            found = 0
            for word in words:
                embedding_vector = embeddings_index.get(word.lower())
                if embedding_vector is not None:
                    found += 1
                    r_vector += embedding_vector
            # if all words of a relation are not in our pretrained glove, set to all-zeros
            if found == 0:
                self.rel_embedding[index] = np.zeros((self.word_emb_dim))
            else:
                self.rel_embedding[index] = r_vector/found

    # hstack vectors of all relations to build Rt
    def init_Rt(self):
        for r in self.rs:
            r_embedding = self.rel_embedding.get(r, [])
            if len(r_embedding) == 0:
                print("relation: " + r + " not in vocab")
                exit
            self.Rt.append(r_embedding)
        self.Rt = np.array(self.Rt)

    # TODO: embed node here
    def embed(self):
        pass

    def get_embedded_state(self):
        return (self.H,self.Rt)

# spliters: split the relation into words
def camel_case_spliter(word):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',word)
    return [m.group(0) for m in matches]

