import torch
import transformers

def init_optimizer(model, config) :
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")
    if optimizer_type == "adam" :
        optimizer = torch.optim.Adam(model.trained_parameters(), lr = learning_rate)
    elif optimizer_type == "sgd" :
        optimizer = torch.optim.SGD(model.trained_parameters(), lr = learning_rate)
    elif optimizer_type == "bert_adam" :
        optimizer = transformers.AdamW(model.trained_parameters(), lr = learning_rate)
    else :
        raise NotImplementedError
    return optimizer