import torch
from transformers import BertForMaskedLM

class MapTuning_MLM(torch.nn.Module) :
    def __init__(self, config) :
        super().__init__()
        try :
            dropout = config.getfloat("model", "dropout")
            self.bert = BertForMaskedLM.from_pretrained(config.get("model", "PLM_path"),
                                        attention_probs_dropout_prob = dropout,
                                        hidden_dropout_prob = dropout)
            self.train()
        except :
            self.bert = BertForMaskedLM.from_pretrained(config.get("model", "PLM_path"))
            self.eval()
    def forward(self, data, bert_embedding, NeedMapper, mapper) :
        tokens, mask, labels = data["tokens"], data["mask"], data["labels"]
        shape_temp = tokens.shape
        tokens = tokens.reshape((-1, ))
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        vecs = torch.cat([(torch.tensor(bert_embedding[id], device = device) if (not NeedMapper[id.item()])
                            else mapper(torch.tensor(bert_embedding[id], device = device))).reshape((+1, -1))
                                            for id in tokens], dim = 0).reshape(shape_temp + (-1, ))
        if device == "cuda" :
            mask, labels = mask.cuda(), labels.cuda()
        return self.bert(inputs_embeds = vecs, attention_mask = mask, labels = labels).loss

from tools.eval_tool import gen_time_str, output_value
from timeit import default_timer as timer
from model.optimizer import init_optimizer
import os
def train(config, mapper, dataset, bert_embedding, NeedMapper, output_path, total_epoch) :
    model = MapTuning_MLM(config)
    if torch.cuda.is_available() :
        model = model.cuda()
    optimizer = init_optimizer(mapper, config)

    total_len = len(dataset)
    for epoch in range(1, total_epoch + 1) :
        total_loss = 0.0
        start_time = timer()
        for step, data in enumerate(dataset) :
            mapper.train()
            optimizer.zero_grad()
            loss = model(data, bert_embedding, NeedMapper, mapper)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            output_loss = total_loss / (step + 1)
            delta_t = timer() - start_time
            output_value(epoch, "train", "{}/{}".format(step + 1, total_len), "{}/{}".format(
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             round(output_loss, 4), "", "\r", config)
        with open(os.path.join(output_path, "loss_{}".format(epoch)), "w") as fout :
            fout.write(str(output_loss))
        model_to_save = mapper.module if hasattr(mapper, "module") else mapper
        model_to_save.save(os.path.join(output_path, "Affine_{}.bin".format(epoch)))