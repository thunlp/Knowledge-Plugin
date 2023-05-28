
import json
import torch
from formatter.basic_formatter import bert_formatter
import torch
import os
from transformers import BertModel
from tqdm import tqdm

class FewRel_submission_formatter(bert_formatter) :
    def __init__(self, config, data_path, data_name, Nway, Kshot) :
        super().__init__(config)
        self.data_path, self.data_name = data_path, data_name
        self.Nway, self.Kshot = Nway, Kshot
    def convert(self, instance) :
        token, pos1, pos2 = instance["tokens"], instance["h"][-1][0], instance["t"][-1][0]
        pos1 = (pos1[0], pos1[-1] + 1)
        pos2 = (pos2[0], pos2[-1] + 1)
        token = [x.lower() for x in token]
        if self.config.get("mapper", "mapper_type") != "None" :
            hid = "hashdonothackme" + instance["h"][1].lower() + "hashdonothackme"
            tid = "hashdonothackme" + instance["t"][1].lower() + "hashdonothackme"
            self.insert_Qid(hid, instance["h"][1])
            self.insert_Qid(tid, instance["t"][1])
            if pos1[0] < pos2[0] :
                token = token[: pos1[0]] + ["#"] + [hid] + ["/"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] : pos2[0]] + ["$"] + [tid] + ["/"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] :]
            else :
                token = token[: pos2[0]] + ["$"] + [tid] + ["/"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] : pos1[0]] + ["#"] + [hid] + ["/"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] :]
        else :
            if pos1[0] < pos2[0] :
                token = token[: pos1[0]] + ["#"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] : pos2[0]] + ["$"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] :]
            else :
                token = token[: pos2[0]] + ["$"] + token[pos2[0] : pos2[1]] + ["$"] + token[pos2[1] : pos1[0]] + ["#"] + token[pos1[0] : pos1[1]] + ["#"] + token[pos1[1] :]
        text = token
        token = []
        for substring in text :
            token += self.tokenizer.tokenize(substring)
        return token
    def concat(self, tokens_1, tokens_2, max_length) :
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
        token = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
        segment = [0] * (1 + len(tokens_1) + 1) + [1] * (len(tokens_2) + 1)
        token = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * len(token)
        assert(self.max_length >= len(token))
        return {"token" : token, "mask" : mask, "segment" : segment}, max(max_length, len(token))
    def _process(self, data) :
        data["meta_test"] = self.convert(data["meta_test"])
        for info in data["meta_train"] :
            for index, s in enumerate(info) :
                info[index] = self.convert(s)
        def instance_merge(instance_1, instance_2) :
            for key in instance_1 :
                if instance_1[key] is None :
                    instance_1[key] = instance_2[key]
                else :
                    instance_1[key] = torch.cat((instance_1[key], instance_2[key]), dim = 0)
        instances = []
        for info in data["meta_train"] :
            temporary, max_length = [], 0
            for s in info :
                instance, max_length = self.concat(data["meta_test"], s, max_length)
                temporary.append(instance)
                instance = {"token" : None, "mask" : None, "segment" : None}
            for _instance in temporary :
                padding = [0] * (max_length - len(_instance["token"]))
                for key in ("token", "mask", "segment") :
                    _instance[key] = torch.LongTensor([_instance[key] + padding])
                instance_merge(instance, _instance)
            instances.append(instance)
        return instances
    def process(self) :
        data_path = os.path.join(self.data_path, "{}-{}-{}.json".format(self.data_name, self.Nway, self.Kshot))
        with open(data_path, "r", encoding = "utf-8") as fin :
            pool = json.load(fin)
        inputs = []
        for data in tqdm(pool) :
            inputs.append(self._process(data))
        return inputs

class FewRel_submission_BERT(torch.nn.Module) :
    def __init__(self, config, Bert_Base = BertModel) :
        super().__init__()
        self.bert = Bert_Base.from_pretrained(config.get("model", "PLM_path"),
                    output_attentions = False,
                    output_hidden_states = False)
        self.fc = torch.nn.Linear(768, 1)
        self.criterion = torch.nn.CrossEntropyLoss()
    def really_forward(self, vecs, mask, segment, shape_temp) :
        y = self.bert(inputs_embeds = vecs, attention_mask = mask, token_type_ids = segment).pooler_output
        y = y.reshape((shape_temp[0], -1))
        y = self.fc(y)
        y = y.reshape((shape_temp[0], ))
        return y
    def forward(self, data, PLM_embedding, NeedMapper, mapper) :
        answer = []
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        for query in tqdm(data) :
            logits = None
            for instance in query :
                token, mask, segment = instance["token"], instance["mask"], instance["segment"]
                shape_temp = token.shape
                token = token.reshape((-1, ))
                vecs = torch.cat([torch.tensor(PLM_embedding[id], device = device) if (not NeedMapper[id.item()])
                        else mapper(torch.tensor(PLM_embedding[id], device = device))
                        for id in token]).reshape(shape_temp + (-1, ))
                if device == "cuda" :
                    mask, segment = mask.cuda(), segment.cuda()
                result = self.really_forward(vecs, mask, segment, shape_temp)
                result = torch.mean(result).reshape((1, 1))
                if logits is None :
                    logits = result
                else :
                    logits = torch.cat((logits, result), dim = 0)
            logits = logits.reshape((-1, ))
            answer.append(logits.argmax().item())
        return answer

import argparse
from config_parser import create_config
from tools.gpu_tool import set_gpu

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required = True)
    parser.add_argument("--gpu", "-g", default = None)
    parser.add_argument("--checkpoint", help = "checkpoint file path", required = True)
    parser.add_argument("--data_path", required = True)
    parser.add_argument("--data_name", required = True)
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    set_gpu(args.gpu)
    
    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    os.makedirs(output_path, exist_ok = True)
    for Nway in (5, 10) :
        for Kshot in (1, 5) :
            Formatter = FewRel_submission_formatter(config, args.data_path, args.data_name, Nway, Kshot)
            model = FewRel_submission_BERT(config)
            model.load_state_dict(torch.load(args.checkpoint)["model"])
            if torch.cuda.is_available() :
                model = model.cuda()
            model.eval()
            with torch.no_grad() :
                answer = model(Formatter.process(), Formatter.get_embedding(), Formatter.get_NeedMapper(), Formatter.get_mapper())
            with open(os.path.join(output_path, "pred-{}-{}.json".format(Nway, Kshot)), "w") as fout :
                json.dump(answer, fout)