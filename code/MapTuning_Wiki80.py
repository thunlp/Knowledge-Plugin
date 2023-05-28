from MapTuning import train
from formatter.basic_formatter import bert_formatter
import json
import torch
import os
from tqdm import tqdm
from tools.seed import seed_everything
from tools.gpu_tool import set_gpu

class MapTuing_Wiki80_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)
    def get_format(self, mention, Qid) :
        Qword = None
        if Qid not in self.keembedding.ent_map :
            Qid = None
        else :
            Qword = "hashdonothackme" + Qid.lower() + "hashdonothackme"
            self.insert_Qid(Qword, Qid)
        MASKS = ["[MASK]"] * len(self.tokenizer.tokenize(" ".join(mention)))
        if Qid is None :
            pass
        else :
            mention = [Qword] + ["/"] + mention
            MASKS = [Qword] + ["/"] + MASKS
        return [" "] + mention + [" "], [" "] + MASKS + [" "], (Qid is not None)
    def template2sentence(self, text, h_word, t_word, pos1, pos2) :
        if pos1[0] < pos2[0] :
            text = text[: pos1[0]] + h_word + text[pos1[1] : pos2[0]] + t_word + text[pos2[1] :]
        else :
            text = text[: pos2[0]] + t_word + text[pos2[1] : pos1[0]] + h_word + text[pos1[1] :]
        return " ".join(text)
    def process(self, mode) :
        data_path = self.config.get("data", f"data_path")
        with open(os.path.join(data_path, "{}.txt".format(mode)), "r", encoding = "utf-8") as fin :
            data = [json.loads(line) for line in fin]
        
        inputs = []
        for instance in tqdm(data) :
            text, pos1, pos2 = instance["token"], instance["h"]["pos"], instance["t"]["pos"]
            text = [x.lower() for x in text]
            h_labels, h_MASKS, h_have = self.get_format(text[pos1[0] : pos1[1]], instance["h"]["id"])
            t_labels, t_MASKS, t_have = self.get_format(text[pos2[0] : pos2[1]], instance["t"]["id"])
            if (not h_have) and (not t_have) :
                continue
            Labels = self.tokenizer.tokenize(self.template2sentence(text, h_labels, t_labels, pos1, pos2))
            if len(Labels) > self.max_length - 2 :
                continue
            for a in (False, True) :
                for b in (False, True) :
                    if a and b :
                        continue
                    labels = Labels.copy()
                    tokens = self.tokenizer.tokenize(self.template2sentence(text, 
                                                                            h_labels if a else h_MASKS, 
                                                                            t_labels if b else t_MASKS,
                                                                            pos1, pos2))

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

    Formatter = MapTuing_Wiki80_formatter(config)
    bert_embedding = Formatter.get_embedding()
    NeedMapper = Formatter.get_NeedMapper()
    mapper = Formatter.get_mapper()

    dataset = Formatter.process("train")
    Formatter = None
    dataset = torch.utils.data.DataLoader(dataset = dataset, batch_size = 64, drop_last = False, shuffle = True)
    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    os.makedirs(output_path, exist_ok = True)
    train(config, mapper, dataset, bert_embedding, NeedMapper, output_path, 15)