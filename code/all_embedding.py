
import numpy as np

class KE_Embedding :
    def __init__(self, config) :
        ke_path = config.get("data", "ke_path")
        self.ent_map = {}
        try :
            ke_type = config.get("data", "ke_type")
        except :
            ke_type = "wikipedia" # the default is "wikipedia"
        self.keembs = np.load(ke_path)
        if ke_type == "wikipedia" :
            with open("../knowledge_embedding/wikipedia/entity2id.txt", "r") as fin :
                fin.readline()
                for line in fin :
                    qid, eid = line.split()
                    self.ent_map[qid] = int(eid)
        elif ke_type == "wikimedia" :
            with open("../knowledge_embedding/wikimedia/entities.tsv", "r") as fin :
                fin.readline()
                for line in fin :
                    eid, cid = line.split()
                    self.ent_map[cid] = int(eid)
        else :
            assert(False)
        self.dimension = len(self.keembs[0])
        self.keembs = list(self.keembs)
    def query_Qid(self, Qid : str) :
        assert(Qid[0] == "Q" and Qid[1 :].isdigit())
        if Qid in self.ent_map :
            word = np.array(self.keembs[self.ent_map[Qid]]).astype("float32")
        else :
            word = np.array([0.] * self.dimension).astype("float32")
        return word, True
    def query_Cid(self, Cid : str) :
        assert(Cid[0] == "C" and Cid[1 :].isdigit())
        if Cid in self.ent_map :
            word = np.array(self.keembs[self.ent_map[Cid]]).astype("float32")
        else :
            word = np.array([0.] * self.dimension).astype("float32")
        return word, True
    def query_id(self, id : int) :
        return np.array(self.keembs[id]).astype("float32"), True

def get_PLMembeddings(model) : # model is in BERT-family
    return list(model.embeddings.word_embeddings.weight.detach().numpy().astype("float32"))