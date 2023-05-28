import os
import json
import torch
from .basic_formatter import PLM_formatter
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
import random
import gc

class FewRel_PLM_formatter(PLM_formatter) :
    def __init__(self, config, BaseModel, BaseTokenizer) :
        super().__init__(config, BaseModel, BaseTokenizer)
        self.data_path = self.config.get("data", "data_path")
        self.pools = {}
        self.N, self.K, self.Q = config.getint("data", "Nway"), config.getint("data", "Kshot"), config.getint("data", "Qquery")
        if self.config.get("mapper", "mapper_type") == "None" :
            return
        necessary = [False] * len(self.keembedding.keembs)
        for mode in ("train", "valid", "test") :
            try :
                with open(os.path.join(self.data_path, self.config.get("data", "{}_file".format(mode))), "r", encoding = "utf-8") as fin :
                    self.pools[mode] = json.load(fin)
                pool = self.pools[mode]
            except :
                continue
            for relation, data in pool.items() :
                for instance in data :
                    for et in ("h", "t") :
                        if instance[et][1] in self.keembedding.ent_map :
                            necessary[self.keembedding.ent_map[instance[et][1]]] = True
        for i in range(len(self.keembedding.keembs)) :
            if not necessary[i] :
                self.keembedding.keembs[i] = None
        gc.collect()
    def Tokenize(self, token) :
        text = token
        token = []
        for substring in text :
            token += self.tokenizer.tokenize(substring)
        return token
    def convert(self, instance) :
        token, pos1, pos2 = instance["tokens"], instance["h"][-1][0], instance["t"][-1][0]
        pos1, pos2 = (pos1[0], pos1[-1] + 1), (pos2[0], pos2[-1] + 1)
        token = [x.lower() for x in token]
        mapper_type = self.config.get("mapper", "mapper_type")
        if mapper_type != "None" :
            assert(instance["h"][1][0] == instance["t"][1][0])
            hid = "hashdonothackme" + instance["h"][1].lower() + "hashdonothackme"
            tid = "hashdonothackme" + instance["t"][1].lower() + "hashdonothackme"
            if instance["h"][1][0] == "Q" :
                self.insert_Qid(hid, instance["h"][1])
                self.insert_Qid(tid, instance["t"][1])
            elif instance["h"][1][0] == "C" :
                self.insert_Cid(hid, instance["h"][1])
                self.insert_Cid(tid, instance["t"][1])
            else :
                assert(False)
            if pos1[0] < pos2[0] :
                token = token[: pos1[0]] + ["#"] + [hid] + ["/"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] : pos2[0]] + ["$"] + [tid] + ["/"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] :]
            else :
                token = token[: pos2[0]] + ["$"] + [tid] + ["/"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] : pos1[0]] + ["#"] + [hid] + ["/"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] :]
        else :
            if pos1[0] < pos2[0] :
                token = token[: pos1[0]] + ["#"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] : pos2[0]] + ["$"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] :]
            else :
                token = token[: pos2[0]] + ["$"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] : pos1[0]] + ["#"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] :]
        result = self.Tokenize(token)
        return result
    def truncate_to_maxlength(self, tokens_1, tokens_2) :
        while 1 + len(tokens_1) + 1 + len(tokens_2) + 1 > self.max_length :
            if tokens_2[0] not in ("$", "#") :
                tokens_2 = tokens_2[1 :]
            elif tokens_2[-1] not in ("$", "#") :
                tokens_2 = tokens_2[: -1]
            elif tokens_1[0] not in ("$", "#") :
                tokens_1 = tokens_1[1 :]
            elif tokens_1[-1] not in ("$", "#") :
                tokens_1 = tokens_1[: -1]
            else :
                assert(False)
        return tokens_1, tokens_2
    def concat(self, tokens_1, tokens_2, max_length) :
        tokens_1, tokens_2 = self.truncate_to_maxlength(tokens_1, tokens_2)
        token = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
        segment = [0] * (1 + len(tokens_1) + 1) + [1] * (len(tokens_2) + 1)
        token = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * len(token)
        assert(self.max_length >= len(token))
        return {"token" : token, "mask" : mask, "segment" : segment}, max(max_length, len(token))
    def _process(self, pool, seed) :
        random.seed(seed)
        classes = random.sample(sorted(list(pool)), self.N)
        data = {}
        for relation in classes :
            L = random.sample(range(len(pool[relation])), self.K + self.Q)
            random.shuffle(L)
            data[relation] = {"supporting" : L[0 : self.K], "query" : L[self.K : self.K + self.Q]}

        tokenized = []
        for relation, info in data.items() :
            supporting = []
            for id in info["supporting"] :
                supporting.append(self.convert(pool[relation][id]))
            query = []
            for id in info["query"] :
                query.append(self.convert(pool[relation][id]))
            tokenized.append({"supporting" : supporting, "query" : query})
        inputs = []
        def instance_merge(instance_1, instance_2) :
            for key in instance_1 :
                if instance_1[key] is None :
                    instance_1[key] = instance_2[key]
                else :
                    instance_1[key] = torch.cat((instance_1[key], instance_2[key]), dim = 0)
        for relation_q, info_q in enumerate(tokenized) :
            for q in info_q["query"] :
                instances = []
                label = relation_q
                for info_s in tokenized :
                    temporary, max_length = [], 0
                    for s in info_s["supporting"] :
                        instance, max_length = self.concat(q, s, max_length)
                        temporary.append(instance)
                    instance = {"token" : None, "mask" : None, "segment" : None}
                    for _instance in temporary :
                        padding = [0] * (max_length - len(_instance["token"]))
                        for key in ("token", "mask", "segment") :
                            _instance[key] = torch.LongTensor([_instance[key] + padding])
                        instance_merge(instance, _instance)
                    instances.append(instance)
                inputs.append({"instances" : instances, "label" : label})
        return inputs
    def process(self, mode, seeds) :
        if mode not in self.pools :
            with open(os.path.join(self.data_path, self.config.get("data", "{}_file".format(mode))), "r", encoding = "utf-8") as fin :
                self.pools[mode] = json.load(fin)
        inputs = []
        pool = self.pools[mode]
        for seed in seeds :
            inputs += self._process(pool, seed)
        return inputs

class FewRel_bert_formatter(FewRel_PLM_formatter) :
    def __init__(self, config) :
        super().__init__(config, BertModel, BertTokenizer)

class FewRel_roberta_formatter(FewRel_PLM_formatter) :
    def __init__(self, config) :
        super().__init__(config, RobertaModel, RobertaTokenizer)
    def insert_Cid(self, word, Cid : str) :
        if self.tokenizer.add_tokens([word]) :
            emb, need = self.keembedding.query_Cid(Cid)
            self.PLM_embedding.append(emb)
            self.NeedMapper.append(need)
    def Tokenize(self, token) :
        return self.tokenizer.tokenize(" ".join(token))
    def concat(self, tokens_1, tokens_2, max_length) :
        tokens_1, tokens_2 = self.truncate_to_maxlength(tokens_1, tokens_2)
        token = ["<s>"] + tokens_1 + ["</s>"] + tokens_2 + ["</s>"]
        segment = [0] * len(token)
        token = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * len(token)
        assert(self.max_length >= len(token))
        return {"token" : token, "mask" : mask, "segment" : segment}, max(max_length, len(token))