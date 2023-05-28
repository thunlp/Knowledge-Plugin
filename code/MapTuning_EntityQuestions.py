from MapTuning import train
from formatter.basic_formatter import bert_formatter
import json
import torch
import os
from tqdm import tqdm
from tools.seed import seed_everything
from tools.gpu_tool import set_gpu

class MapTuing_WikiET_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)
        self.mention2Qid = {}
        with open("../knowledge_embedding/wikipedia/entity_map.txt") as fin :
            for line in fin :
                input = line.strip().split("\t")
                self.mention2Qid[input[0]] = input[-1]
        with open(os.path.join(self.config.get("data", "data_path"), "relation_query_templates.json")) as fin :
            self.relation_query_templates = json.load(fin)
    def get_format(self, mention, Qid) :
        MASKS = " ".join(["[MASK]"] * len(self.tokenizer.tokenize(mention)))
        Qword = "hashdonothackme" + Qid.lower() + "hashdonothackme"
        self.insert_Qid(Qword, Qid)
        mention = Qword + " / " + mention
        MASKS = Qword + " / " + MASKS
        return " " + mention + " ", " " + MASKS + " "
    def template2sentence(self, text, word, start, end) :
        return text[: start] + " " + word + " " + text[end :]
    def process(self, mode) :
        data_path = self.config.get("data", f"data_path")
        
        inputs = []
        for relation, template in tqdm(self.relation_query_templates.items()) :
            if not os.path.exists(os.path.join(data_path, mode, "{}.{}.json".format(relation, mode))) :
                continue
            with open(os.path.join(data_path, mode, "{}.{}.json".format(relation, mode))) as fin :
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
                labels, MASKS = self.get_format(text[start : end], Qid)
                labels = self.tokenizer.tokenize(self.template2sentence(text, labels, start, end))
                if len(labels) > self.max_length - 2 :
                    continue
                tokens = self.tokenizer.tokenize(self.template2sentence(text, MASKS, start, end))
                for (i, _token) in enumerate(tokens) :
                    if _token != "[MASK]" :
                        assert(_token == labels[i])
                        labels[i] = -100
                    else :
                        labels[i] = self.tokenizer.vocab[labels[i]]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                labels = [-100] + labels + [-100]
                assert(len(tokens) == len(labels))
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                mask = [1] * len(tokens)
                padding = [0] * (self.max_length - len(tokens))
                labels += [-100] * (self.max_length - len(tokens))
                tokens += padding
                mask += padding
                assert(len(tokens) == len(labels))
                inputs.append({"tokens" : torch.LongTensor(tokens),
                                "mask" : torch.LongTensor(mask),
                                "labels" : torch.LongTensor(labels)})
        return inputs

import argparse
from config_parser import create_config

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required = True)
    parser.add_argument("--gpu", "-g", default = None)
    parser.add_argument("--seed", type = int, default = 64)
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    set_gpu(args.gpu)
    seed_everything(args.seed)

    Formatter = MapTuing_WikiET_formatter(config)
    bert_embedding = Formatter.get_embedding()
    NeedMapper = Formatter.get_NeedMapper()
    mapper = Formatter.get_mapper()

    dataset = Formatter.process("train")
    Formatter = None
    dataset = torch.utils.data.DataLoader(dataset = dataset, batch_size = 64, drop_last = False, shuffle = True)
    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    os.makedirs(output_path, exist_ok = True)
    train(config, mapper, dataset, bert_embedding, NeedMapper, output_path, 5)