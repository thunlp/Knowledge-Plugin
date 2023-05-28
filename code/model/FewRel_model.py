import torch
from transformers import BertModel, RobertaModel
from model import get_trained_parameters, LoRALinear_replace
from .adapter_bert.src.transformers import BertModel as adapter_BertModel
from all_embedding import get_PLMembeddings
from tools.accuracy_init import init_accuracy_function

class FewRel_PLM(torch.nn.Module) :
    def __init__(self, config, BackboneModel) :
        super().__init__()
        try :
            dropout = config.getfloat("model", "dropout")
            self.bert = BackboneModel.from_pretrained(config.get("model", "PLM_path"),
                        output_attentions = False,
                        output_hidden_states = False,
                        attention_probs_dropout_prob = dropout,
                        hidden_dropout_prob = dropout)
        except :
            self.bert = BackboneModel.from_pretrained(config.get("model", "PLM_path"),
                        output_attentions = False,
                        output_hidden_states = False)
        self.fc = torch.nn.Linear(768, 1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config)
        self.original_vocab_size = len(get_PLMembeddings(self.bert))
    def really_forward(self, vecs, mask, segment) :
        y = self.bert(inputs_embeds = vecs, attention_mask = mask, token_type_ids = segment).pooler_output
        y = self.fc(y)
        y = y.squeeze(-1)
        return y
    def forward(self, data, config, acc_result, PLM_embedding, NeedMapper, mapper, mode) :
        Loss = 0.0
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        for query in data :
            logits = None
            for instance in query["instances"] :
                token, mask, segment = instance["token"], instance["mask"], instance["segment"]
                shape_temp = token.shape
                token = token.reshape((-1, ))
                vecs = torch.cat([torch.tensor(PLM_embedding[id], device = device) if (not NeedMapper[id.item()])
                        else mapper(torch.tensor(PLM_embedding[id], device = device))
                        for id in token]).reshape(shape_temp + (-1, ))
                if device == "cuda" :
                    mask, segment = mask.cuda(), segment.cuda()
                result = self.really_forward(vecs, mask, segment)
                result = torch.mean(result).reshape((1, 1))
                if logits is None :
                    logits = result
                else :
                    logits = torch.cat((logits, result), dim = 0)
            logits = logits.reshape((1, ) + logits.shape)
            label = torch.LongTensor([[query["label"]]])
            acc_result = self.accuracy_function(logits.cpu(), label, config, acc_result)
            if device == "cuda" :
                label = label.cuda()
            loss = self.criterion(logits, label)
            if mode == "train" :
                loss.backward()
            Loss += loss.item()
            logits = loss = None
        return {"loss" : Loss, "acc_result" : acc_result}
    def trained_parameters(self) :
        return self.parameters()

class FewRel_BERT(FewRel_PLM) :
    def __init__(self, config) :
        super().__init__(config, BertModel)

class FewRel_RoBERTa(FewRel_PLM) :
    def __init__(self, config) :
        super().__init__(config, RobertaModel)

class FewRel_Adapter(FewRel_PLM) :
    def __init__(self, config) :
        super().__init__(config, adapter_BertModel)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "adapter")
        for (name, param) in self.named_parameters() :
            if "classifier" in name :
                trained_params.append(param)
        return trained_params

class FewRel_LoRA(FewRel_PLM) :
    def __init__(self, config) :
        super().__init__(config, BertModel)
        LoRALinear_replace(self.bert.encoder, 32, self.bert.config, True)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "lora")
        for (name, param) in self.named_parameters() :
            if "classifier" in name :
                trained_params.append(param)
        return trained_params

class FewRel_BitFit(FewRel_PLM) :
    def __init__(self, config) :
        super().__init__(config, BertModel)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "bias")
        for (name, param) in self.named_parameters() :
            if ("classifier" in name) and ("bias" not in name) :
                trained_params.append(param)
        return trained_params