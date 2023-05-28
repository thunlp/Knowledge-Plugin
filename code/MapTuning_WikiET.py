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
    def get_format(self, mention, id) :
        MASKS = " ".join(["[MASK]"] * len(self.tokenizer.tokenize(mention)))
        Qword = "hashdonothackme" + id + "hashdonothackme"
        self.insert_id(Qword, int(id))
        mention = Qword + " / " + mention
        MASKS = Qword + " / " + MASKS
        return " " + mention + " ", " " + MASKS + " "
    def template2sentence(self, text, word, start, end) :
        return text[: start] + " " + word + " " + text[end :]
    def process(self, mode) :
        data_path = self.config.get("data", f"data_path")
        with open(os.path.join(data_path, "{}.json".format(mode)), "r", encoding = "utf-8") as fin :
            data = [json.loads(line) for line in fin]
        
        inputs = []
        for instance in tqdm(data) :
            text, start, end = instance["sent"], instance["start"], instance["end"]
            text = text.lower()
            labels, MASKS = self.get_format(text[start : end], instance["id"])
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
    train(config, mapper, dataset, bert_embedding, NeedMapper, output_path, 2)