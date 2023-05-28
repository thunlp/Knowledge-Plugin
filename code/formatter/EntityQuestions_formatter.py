import os
import json
from tqdm import tqdm
import torch
from .basic_formatter import bert_formatter

class EntityQuestions_bert_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)
        self.mention2Qid = {}
        with open("../knowledge_embedding/wikipedia/entity_map.txt") as fin :
            for line in fin :
                input = line.strip().split("\t")
                self.mention2Qid[input[0]] = input[-1]
        with open(os.path.join(self.config.get("data", "data_path"), "relation_query_templates.json")) as fin :
            self.relation_query_templates = json.load(fin)
    def process(self, mode) :
        data_path = self.config.get("data", "data_path")
        mode_file = self.config.get("data", "{}_file".format(mode))
        mapper_type = self.config.get("mapper", "mapper_type")

        contain = 0
        instances = []
        for relation, template in tqdm(self.relation_query_templates.items()) :
            if not os.path.exists(os.path.join(data_path, mode_file, "{}.{}.json".format(relation, mode_file))) :
                continue
            with open(os.path.join(data_path, mode_file, "{}.{}.json".format(relation, mode_file))) as fin :
                data = json.load(fin)
            start = template.find("[X]")
            assert(start >= 0)
            end = (start + 3) - len(template)
            for instance in data :
                text = instance["question"]
                mention = text[start : end]
                if mention not in self.mention2Qid :
                    continue
                Qid = self.mention2Qid[mention]
                text = text.lower()
                label = None
                for answer in instance["answers"] :
                    answer = answer.lower()
                    if answer in self.tokenizer.vocab :
                        label = self.tokenizer.vocab[answer]
                        break
                if label is None :
                    continue
                if mapper_type != "None" :
                    id = "hashdonothackme" + Qid.lower() + "hashdonothackme"
                    self.insert_Qid(id, Qid)
                    text = text[: start] + "$ " + id + " / " + text[start : end] + " $" + text[end :]
                else :
                    text = text[: start] + "$ " + text[start : end] + " $" + text[end :]
                text += " [MASK]"
                token, mask = self.text_to_input(text)
                mask_position = None
                for (i, id) in enumerate(token) :
                    if self.tokenizer.ids_to_tokens[id] == "[MASK]" :
                        assert(mask_position is None)
                        mask_position = i
                assert(mask_position is not None)
                instances.append({"input" : {"token" : torch.LongTensor(token), "mask" : torch.LongTensor(mask)}, "label" : {"label" : label, "mask_position" : mask_position}})
        print("{} / {}".format(contain, len(instances)))
        return instances