import json

class KnowledgeGraph:
    def __init__(self, file_path, vocab_path):
        self.rel_vocab = None
        self.en_vocab = None
        self.text_graph = {}
        self.graph = {}       # encoded graph (using index in vocab)
        self.inv_graph = {}   # (r, e2) -> e1

        self.read_vocab(vocab_path)
        self.read_graph(file_path)

    # read text graph from file
    def read_graph(self, file_path):
        with open(file_path, 'rb') as fp:
            line = fp.readline()
            while line:
                e1, r, e2 = line[0:-2].decode("utf-8").split("\t")
                # update the text_graph (not include reverse)
                if e2 not in self.text_graph:
                    self.text_graph[e2] = []
                self.text_graph[e2].append((r,e1))

                # update graph
                ie1, ir, ie2 = self.encode_edge((e1, r, e2))
                inv_ir = self.rel_vocab["inv_" + r]
                if ie2 not in self.graph:
                    self.graph[ie2] = []
                if ie1 not in self.graph:
                    self.graph[ie1] = []
                self.graph[ie2].append((ir,ie1))
                self.graph[ie1].append((inv_ir, ie2))

                # update inv_graph
                if (ir, ie1) not in self.inv_graph:
                    self.inv_graph[(ir, ie1)] = []
                if (inv_ir,ie2) not in self.inv_graph:
                    self.inv_graph[(inv_ir,ie2)] = []
                self.inv_graph[(ir, ie1)].append(ie2)
                self.inv_graph[(inv_ir,ie2)].append(ie1)

                line = fp.readline()
            
            # add inverse relation to graph and inv_graph
            self_loop_relation = self.rel_vocab["NO_OP"]
            for e in self.en_vocab:
                ie = self.en_vocab[e]
                if ie not in self.graph:
                    self.graph[ie] = []
                if (self_loop_relation, ie) not in self.inv_graph:
                    self.inv_graph[(self_loop_relation, ie)] = []
                self.graph[ie].append((self_loop_relation,ie))
                self.inv_graph[(self_loop_relation,ie)].append(ie)
 
    # read vocab from file
    def read_vocab(self, vocab_path):
        with open(vocab_path + "entity_vocab.json") as json_file:
            self.en_vocab = json.load(json_file)
        with open(vocab_path + "relation_vocab.json") as json_file:
            self.rel_vocab = json.load(json_file)
            size = len(self.rel_vocab)
            inv = {}
            for r in self.rel_vocab:
                if r != "NO_OP" and r != "DUMMY_START_RELATION":
                    inv["inv_" + r] = self.rel_vocab[r] + size - 3
            for r in inv:
                self.rel_vocab[r] = inv[r]


    # encode an edge
    def encode_edge(self, edge):
        e1, r, e2 = edge
        index_e1 = self.en_vocab[e1]
        index_e2 = self.en_vocab[e2]
        index_r  = self.rel_vocab[r]
        return index_e1, index_r, index_e2

    # encode a question of format ([e1,e2 ...], [r1, r2 ...])
    def encode_question(self, question):
        ens, rels = question
        iens = []
        irels = []
        for e in ens:
            iens.append(self.en_vocab[e])
        for r in rels:
            irels.append(self.rel_vocab[r])
        return (iens, irels)

