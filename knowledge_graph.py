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
                e2, r, e1 = line[0:-1].decode("utf-8").split("\t")

                # for countries
                if e1[-1] == '\r':
                    e1 = e1[0:-1]
                if e2[-1] == "\r":
                    e2 = e2[0:-1]

                # update the text_graph
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

    def encode_answers(self, answers):
        res = []
        for answer in answers:
            res.append(self.en_vocab[answer])
        return res
    
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

    def max_shortest_path(self, nodes):
        max_distance = None
        for s in nodes:
            for e in nodes:
                if s != e:
                    dist = self.shortest_path(s,e)
                    if max_distance == None or max_distance < dist:
                        max_distance = dist
        if max_distance == None:
            max_distance = 0
        return max_distance

    def shortest_path(self, s, e):
        q = []
        visited = {}
        q.append(s)
        visited[s] = 0 
        while len(q) != 0:
            cur = q.pop()
            if visited[cur] > 4:
                return 4
            for r, node in self.graph[cur]:
                if node == e:
                    return visited[cur] + 1
                if node not in visited:
                    q.append(node)
                    visited[node] = visited[cur] + 1

        return 4


