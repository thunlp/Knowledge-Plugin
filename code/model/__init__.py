import torch
from tools.accuracy_init import init_accuracy_function

class general_model(torch.nn.Module) :
    def __init__(self, config) :
        super().__init__()
        self.accuracy_function = init_accuracy_function(config)
    def forward(self, data, config, acc_result, bert_embedding, NeedMapper, mapper, mode) :
        inputs = data["input"]
        token, mask = inputs["token"], inputs["mask"]
        shape_temp = token.shape
        token = token.reshape((-1, ))
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        vecs = torch.cat([torch.tensor(bert_embedding[id], device = device) if (not NeedMapper[id.item()])
                        else mapper(torch.tensor(bert_embedding[id], device = device))
                        for id in token]).reshape(shape_temp + (-1, ))
        if device == "cuda" :
            mask = mask.cuda()
        return self.forward_calc(config, acc_result, vecs, mask, data["label"])
    def trained_parameters(self) :
        return self.parameters()

def get_trained_parameters(model, substr : str) :
    trained_params = []
    for (name, param) in model.named_parameters() :
        if name.startswith("bert") :
            if substr in name :
                trained_params.append(param)
        else :
            trained_params.append(param)
    return trained_params

# LoRA lib
import math

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = torch.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(torch.nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        torch.nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = torch.nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = torch.nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        torch.nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        torch.nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        torch.nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = torch.nn.functional.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return torch.nn.functional.linear(x, T(self.weight), bias=self.bias)

def LoRALinear_replace(model, lora_r : int, config, replace_BertIntermediateDense : bool) : # model is BertEncoder
    num_attention_heads = config.num_attention_heads
    attention_head_size = int(config.hidden_size / config.num_attention_heads)
    all_head_size = num_attention_heads * attention_head_size
    for layer in range(config.num_hidden_layers) :
        temp_query, temp_key = model.layer[layer].attention.self.query, model.layer[layer].attention.self.key
        model.layer[layer].attention.self.query = Linear(config.hidden_size, all_head_size, r = lora_r)
        model.layer[layer].attention.self.key = Linear(config.hidden_size, all_head_size, r = lora_r)
        model.layer[layer].attention.self.query.weight.data = temp_query.weight.data
        model.layer[layer].attention.self.key.weight.data = temp_key.weight.data
        if replace_BertIntermediateDense :
            temp_dense = model.layer[layer].intermediate.dense
            model.layer[layer].intermediate.dense = Linear(config.hidden_size, config.intermediate_size, r = lora_r)
            model.layer[layer].intermediate.dense.weight.data = temp_dense.weight.data