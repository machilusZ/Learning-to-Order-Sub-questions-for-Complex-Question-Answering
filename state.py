import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import torch
import re

class State:
    def __init__(self, question, graph, word_emb_dim):
        self.es, self.rs = graph.encode_question(question) # encoded entities and relations in the question
        self.word_emb_dim = word_emb_dim  # dimension of word embeding
        self.subgraphs = []     # each element is a vector of format [e1, e2, ... ] representing a subgraph
        self.rel_embedding = {} # mapping from all the relations in vocb to its embeding
        self.node_embedding = {} # mappin from all nodes to its embeding
        self.Rt = []            # Rt in the state (each row is a embedded relation)
        self.Ht  = []           # Ht in the state
        self.graph = graph      # knowledge graph object

        # init all subgraphs from the question
        for e in self.es:
            self.subgraphs.append([e])

        # get word vectors for each relations
        self.init_rel_embedding("glove.840B.300d.txt",camel_case_spliter)

        # get node embeding for each entity
        self.init_node_embedding("./data/countries/countries_embed.npy")

        # init Rt from the relations in the question
        self.init_Rt()

        # init Ht
        self.init_Ht()

    # add a node to one of the subgraph, input format (r, e2)
    def update(self, action):
        # add new node to the graph
        r, e2 = action
        done = False
        for subgraph in self.subgraphs:
            for e in subgraph:
                if e in self.graph.inv_graph[action]:
                    subgraph.append(e2)
                    done = True
                    break
            if done:
                break

        # update Rt
        rt_embed = self.rel_embedding[r]
        for i in range(self.Rt.shape[0]):
            # if Rt is already all zeros, we will not reduce it
            if np.sum(self.Rt[i] != 0) != 0:
                gamma = 1 - cosine(self.Rt[i], rt_embed)
                self.Rt[i] -= gamma * rt_embed

        # TODO: update Ht


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
            neighbors = self.graph.graph.get(e1, [])
            for (r, e2) in neighbors:
                if e2 not in subgraph:
                    ret.append((e1, r, e2))
        return ret

    # generate all possible actions (r, e) according given all the current neighbors
    def generate_all_possible_actions(self):
        neighbors = self.find_all_neighbors()
        actions = []
        for (_, r, e2) in neighbors:
            if (r, e2) not in actions:
                actions.append((r, e2))
        return actions

    # embed relations
    def init_rel_embedding(self, path_to_embedding, spliter):
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
                self.rel_embedding[index] = np.random.randn((self.word_emb_dim))
            else:
                self.rel_embedding[index] = r_vector/found

    # embed nodes: change this
    def init_node_embedding(self, path):
        # load pretrained embedding directly
        self.node_embedding = np.load(path)

    # get vectors of all relations to build Rt
    def init_Rt(self):
        for r in self.rs:
            r_embedding = self.rel_embedding.get(r, [])
            if len(r_embedding) == 0:
                print("relation: " + r + " not in vocab")
                exit
            self.Rt.append(r_embedding)
        self.Rt = np.array(self.Rt)

    # TODO: change this
    def init_Ht(self):
        for subgraph in self.subgraphs:
            e = subgraph[0]
            self.Ht.append(self.node_embedding[e])

    def get_embedded_state(self):
        return (torch.tensor(self.Ht).float(),torch.tensor(self.Rt).float())

    def get_input_size(self):
        Ht_len = torch.tensor(self.Ht).view(-1).size()[0]
        Rt_len = torch.tensor(self.Rt).view(-1).size()[0]
        return Ht_len + Rt_len

# spliters: split the relation into words
def camel_case_spliter(word):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',word)
    return [m.group(0) for m in matches]

