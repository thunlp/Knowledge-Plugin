import os
import json
from tqdm import tqdm
import torch
from .basic_formatter import bert_formatter

class Wiki80_bert_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)
        with open(os.path.join(config.get("data", "data_path"), "rel2id.json"), "r") as fin:
            self.rel_map = json.loads(fin.read())
    def process(self, mode) :
        data_path = self.config.get("data", "data_path")
        mode_file = self.config.get("data", "{}_file".format(mode))
        with open(os.path.join(data_path, mode_file), "r", encoding = "utf-8") as fin :
            data = [json.loads(line) for line in fin]
        
        instances = []
        mapper_type = self.config.get("mapper", "mapper_type")
        for instance in tqdm(data) :
            token, pos1, pos2 = instance["token"], instance["h"]["pos"], instance["t"]["pos"]
            token = [x.lower() for x in token]
            if mapper_type != "None" :
                hid = "hashdonothackme" + instance["h"]["id"].lower() + "hashdonothackme"
                tid = "hashdonothackme" + instance["t"]["id"].lower() + "hashdonothackme"
                self.insert_Qid(hid, instance["h"]["id"])
                self.insert_Qid(tid, instance["t"]["id"])
                if pos1[0] < pos2[0] :
                    token = token[: pos1[0]] + ["#"] + [hid] + ["/"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] : pos2[0]] + ["$"] + [tid] + ["/"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] :]
                else :
                    token = token[: pos2[0]] + ["$"] + [tid] + ["/"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] : pos1[0]] + ["#"] + [hid] + ["/"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] :]
            else :
                if pos1[0] < pos2[0] :
                    token = token[: pos1[0]] + ["#"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] : pos2[0]] + ["$"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] :]
                else :
                    token = token[: pos2[0]] + ["$"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] : pos1[0]] + ["#"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] :]
            text = " ".join(token)
            token, mask = self.text_to_input(text)
            instances.append({"input" : {"token" : torch.LongTensor(token), "mask" : torch.LongTensor(mask)},
                            "label" : self.rel_map[instance["relation"]]})
        return instances