import torch
from transformers import BertForMaskedLM
from model import general_model, get_trained_parameters, LoRALinear_replace
from .adapter_bert.src.transformers import BertForMaskedLM as adapter_BertForMaskedLM
from all_embedding import get_PLMembeddings

class EntityQuestions_BERT(general_model) :
    def __init__(self, config, BackboneModel = BertForMaskedLM) :
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
        label, mask_position = label["label"], label["mask_position"]
        y = self.bert(inputs_embeds = vecs, attention_mask = mask).logits
        y = torch.stack([y[i][p.item()] for (i, p) in enumerate(mask_position)])
        acc_result = self.accuracy_function(y.cpu(), label, config, acc_result)
        if torch.cuda.is_available() :
            label = label.cuda()
        loss = self.criterion(y, label)
        return {"loss" : loss, "acc_result" : acc_result}

class EntityQuestions_Adapter(EntityQuestions_BERT) :
    def __init__(self, config) :
        super().__init__(config, adapter_BertForMaskedLM)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "adapter")
        return trained_params

class EntityQuestions_LoRA(EntityQuestions_BERT) :
    def __init__(self, config) :
        super().__init__(config, BertForMaskedLM)
        LoRALinear_replace(self.bert.bert.encoder, 4, self.bert.config, True)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "lora")
        return trained_params

class EntityQuestions_BitFit(EntityQuestions_BERT) :
    def __init__(self, config) :
        super().__init__(config)
    def trained_parameters(self) :
        trained_params = get_trained_parameters(self, "bias")
        return trained_params