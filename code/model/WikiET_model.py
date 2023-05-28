import torch
from transformers import BertModel
from model import general_model, get_trained_parameters, LoRALinear_replace
from .adapter_bert.src.transformers import BertModel as adapter_BertModel
from all_embedding import get_PLMembeddings

class WikiET_BERT(general_model) :
    def __init__(self, config, BackboneModel = BertModel) :
        super().__init__(config)
        num_labels = config.getint("model", "output_dim")
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
        self.fc = torch.nn.Linear(768, num_labels)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.original_vocab_size = len(get_PLMembeddings(self.bert))
    def forward_calc(self, config, acc_result, vecs, mask, label) :
        y = self.bert(inputs_embeds = vecs, attention_mask = mask).pooler_output
        y = self.fc(y)

        acc_result = self.accuracy_function(y.cpu(), label, config, acc_result)
        if torch.cuda.is_available() :
            label = label.cuda()
        loss = self.criterion(y, label.float())
        return {"loss" : loss, "acc_result" : acc_result}

class WikiET_Adapter(WikiET_BERT) :
    def __init__(self, config) :
        super().__init__(config, adapter_BertModel)
    def trained_parameters(self) :
        return get_trained_parameters(self, "adapter")

class WikiET_LoRA(WikiET_BERT) :
    def __init__(self, config) :
        super().__init__(config, BertModel)
        LoRALinear_replace(self.bert.encoder, 4, self.bert.config, False)
    def trained_parameters(self) :
        return get_trained_parameters(self, "lora")

class WikiET_BitFit(WikiET_BERT) :
    def __init__(self, config) :
        super().__init__(config)
    def trained_parameters(self) :
        return get_trained_parameters(self, "bias")