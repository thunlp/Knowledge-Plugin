import os
import json
from tqdm import tqdm
import torch
from .basic_formatter import bert_formatter

class WikiET_bert_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)
        with open(os.path.join(config.get("data", "data_path"), "label2id.json"), "r") as fin :
            self.label_map = json.loads(fin.read())
    def process(self, mode) :
        data_path = self.config.get("data", "data_path")
        mode_file = self.config.get("data", "{}_file".format(mode))
        with open(os.path.join(data_path, mode_file), "r", encoding = "utf-8") as fin :
            data = [json.loads(line) for line in fin]
        
        instances = []
        mapper_type = self.config.get("mapper", "mapper_type")
        for instance in tqdm(data) :
            text, start, end = instance["sent"], instance["start"], instance["end"]
            text = text.lower()
            if mapper_type != "None" :
                id = "hashdonothackme" + instance["id"] + "hashdonothackme"
                self.insert_id(id, int(instance["id"]))
                text = text[: start] + "$ " + id + " / " + text[start : end] + " $" + text[end :]
            else :
                text = text[: start] + "$ " + text[start : end] + " $" + text[end :]
            token, mask = self.text_to_input(text)
            label = torch.zeros(size = (len(self.label_map), ), dtype = int)
            for l in instance["labels"] :
                label[self.label_map[l]] = 1
            instances.append({"input" : {"token" : torch.LongTensor(token), "mask" : torch.LongTensor(mask)},
                            "label" : label})
        return instances