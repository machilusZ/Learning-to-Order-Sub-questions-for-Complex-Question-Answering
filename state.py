import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import torch.nn as nn
import torch
import re

class State:
    def __init__(self, question, graph, word_emb_dim, word2node, attention, rel_embedding, T, device):
        self.es, self.rs = graph.encode_question(question) # encoded entities and relations in the question
        self.word_emb_size = word_emb_dim  # dimension of word embeding
        self.subgraphs = []                # each element is a vector of format [e1, e2, ... ] representing a subgraph
        self.rel_embedding = rel_embedding # mapping from all the relations in vocb to its embeding
        self.node_embedding_size = None    # size of node embeding
        self.node_embedding = {}           # mappin from all nodes to its embeding
        self.Rt = []                       # Rt in the state (each row is a embedded relation)
        self.ht = []                       # hti for each of the subgraph
        self.Ht  = []                      # Ht in the state
        self.graph = graph                 # knowledge graph object
        self.word2node = word2node         # a fc layer project word embeding to node embedingg
        self.attention = attention         # mutihead self-attention 
        self.T = T                         # T step
        self.device = device

        # init all subgraphs from the question
        for e in self.es:
            self.subgraphs.append([e])

        # get node embeding for each entity
        self.init_node_embedding("./data/wc/wc_embed.npy")

        # init Rt from the relations in the question
        self.init_Rt()

        # init Ht
        self.calculate_ht()
        self.calculate_Ht()

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
                self.Rt[i] -= (gamma * rt_embed)/self.T

        # update Ht
        self.calculate_ht()
        self.calculate_Ht()

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

    # calculate ht based on self.subgraphs
    def calculate_ht(self):
        projected_Rt = torch.t(self.word2node(torch.Tensor(self.Rt).to(self.device)))
        init_ht = False
        if len(self.ht) == 0:
            init_ht = True

        for i, subgraph in enumerate(self.subgraphs):
            gti = []
            for e in subgraph:
                gti.append(self.node_embedding[e])
            gti = torch.Tensor(gti)
            if init_ht:
                self.ht.append(get_hti(gti, projected_Rt))
            else:
                self.ht[i] = get_hti(gti, projected_Rt)

    def calculate_Ht(self):
        init_Ht = False
        if len(self.Ht) == 0:
            init_Ht = True
        for i, hti in enumerate(self.ht):
            Hti = self.attention(torch.t(hti).to(self.device))
            if init_Ht:
                self.Ht.append(Hti)
            else:
                self.Ht[i] = Hti

    def get_embedded_state(self):
        return (self.Ht,torch.Tensor(self.Rt).float())

    def get_input_size(self):
        Ht_len = torch.stack(self.Ht).view(-1).size()[0]
        Rt_len = torch.Tensor(self.Rt).view(-1).size()[0]
        return Ht_len + Rt_len


# helper function for calculate hti
# shape of gti (k+1, embedding_size), shape of Rt(embedding_size, m + 1)
def get_hti(gti, Rt):
    L = torch.mm(gti,Rt)
    A_Rt = F.softmax(L, dim=1)
    hti = torch.mm(torch.t(gti),A_Rt)

    return hti

