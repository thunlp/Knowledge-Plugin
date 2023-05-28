import torch
from transformers import BertForSequenceClassification
from model import general_model, get_trained_parameters, LoRALinear_replace
from .adapter_bert.src.transformers import BertForSequenceClassification as adapter_BertForSequenceClassification
from all_embedding import get_PLMembeddings

class Wiki80_BERT(general_model) :
    def __init__(self, config, BackboneModel = BertForSequenceClassification) :
        super().__init__(config)
        try :
            dropout = config.getfloat("model", "dropout")
            self.bert = BackboneModel.from_pretrained(config.get("model", "PLM_path"),
                        num_labels = config.getint("model", "output_dim"),
                        output_attentions = False,
                        output_hidden_states = False,
                        attention_probs_dropout_prob = dropout,
                        hidden_dropout_prob = dropout)
        except :
            self.bert = BackboneModel.from_pretrained(config.get("model", "PLM_path"),
                        num_labels = config.getint("model", "output_dim"),
                        output_attentions = False,
                        output_hidden_states = False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.original_vocab_size = len(get_PLMembeddings(self.bert.bert))
    def forward_calc(self, config, acc_result, vecs, mask, label) :
        y = self.bert(inputs_embeds = vecs, attention_mask = mask).logits
        acc_result = self.accuracy_function(y.cpu(), label, config, acc_result)
        if torch.cuda.is_available() :
            label = label.cuda()
        loss = self.criterion(y, label)
        return {"loss" : loss, "acc_result" : acc_result}

class Wiki80_Adapter(Wiki80_BERT) :
    def __init__(self, config) :
        super().__init__(config, adapter_BertForSequenceClassification)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "adapter")
        for (name, param) in self.named_parameters() :
            if "classifier" in name :
                trained_params.append(param)
        return trained_params

class Wiki80_LoRA(Wiki80_BERT) :
    def __init__(self, config) :
        super().__init__(config, BertForSequenceClassification)
        LoRALinear_replace(self.bert.bert.encoder, 32, self.bert.config, True)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "lora")
        for (name, param) in self.named_parameters() :
            if "classifier" in name :
                trained_params.append(param)
        return trained_params

class Wiki80_BitFit(Wiki80_BERT) :
    def __init__(self, config) :
        super().__init__(config)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "bias")
        for (name, param) in self.named_parameters() :
            if ("classifier" in name) and ("bias" not in name) :
                trained_params.append(param)
        return trained_params