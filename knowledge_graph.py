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
                # update the text_graph (! inverse relation here)
                if e2 not in self.text_graph:
                    self.text_graph[e2] = []
                self.text_graph[e2].append((r,e1))

                # update graph
                ie1, ir, ie2 = self.encode_edge((e1, r, e2))
                if ie2 not in self.graph:
                    self.graph[ie2] = []
                self.graph[ie2].append((ir,ie1))

                # update inv_graph
                if (ir, ie1) not in self.inv_graph:
                    self.inv_graph[(ir, ie1)] = []
                self.inv_graph[(ir, ie1)].append(ie2)

                line = fp.readline()

    # read vocab from file
    def read_vocab(self, vocab_path):
        with open(vocab_path + "entity_vocab.json") as json_file:
            self.en_vocab = json.load(json_file)
        with open(vocab_path + "relation_vocab.json") as json_file:
            self.rel_vocab = json.load(json_file)

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

