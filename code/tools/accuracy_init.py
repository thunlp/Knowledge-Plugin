import logging
import torch

logger = logging.Logger(__name__)

def get_prf(res) :
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0 :
        if res["FP"] == 0 and res["FN"] == 0 :
            precision, recall, f1 = 1.0, 1.0, 1.0
        else :
            precision, recall, f1 = 0.0, 0.0, 0.0
    else :
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def gen_micro_macro_result(res) :
    if "class" in res :
        res = res["class"]
    precision, recall, f1 = [], [], []
    total = {"TP" : 0, "FP" : 0, "FN" : 0, "TN" : 0}
    for a in range(len(res)) :
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]
        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision, macro_recall, macro_f1 = 0, 0, 0
    for a in range(len(f1)) :
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]
    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3)
            }

def single_label_top1_accuracy(outputs : torch.Tensor, label : torch.Tensor, config, result = None) :
    if result is None :
        result = {"inst" : [], "class" : [], "total_acc" : 0}
    id1, id2 = outputs.max(dim = 1)[1], label
    nr_classes = outputs.shape[1]
    while len(result["class"]) < nr_classes :
        result["class"].append({"TP": 0, "FN": 0, "FP" : 0, "TN" : 0})
    for a in range(len(id1)) :
        it_is, should_be = id1[a].item(), id2[a].item()
        if it_is == should_be :
            result["class"][it_is]["TP"] += 1
            result["total_acc"] += 1
        else :
            result["class"][it_is]["FP"] += 1
            result["class"][should_be]["FN"] += 1
        result["inst"].append([it_is, should_be])
    return result

def multi_label_accuracy(outputs : torch.Tensor, labels : torch.Tensor, config, result = None) :
    if labels.shape[0] != outputs.shape[0] :
        raise ValueError("Input dimensions of labels and outputs must match.")
    if result is None :
        result = {"inst" : [], "class" : [], "total_acc" : 0}
    total = 0
    nr_classes = outputs.shape[1]
    while len(result["class"]) < nr_classes :
        result["class"].append({"TP": 0, "FN": 0, "FP" : 0, "TN" : 0})

    outputs_res, labels_res = [], []

    for j in range(outputs.shape[0]) :
        threshold = min(0, outputs[j, :].max())
        outputs1 = (outputs[j, :] >= threshold).long()
        labels1 = (labels[j, :].float() >= 0.5).long()
        result["inst"].append([outputs1, labels1])
        outputs_res.append(outputs1)
        labels_res.append(labels1)
        if ((labels1 * outputs1).sum()).item() == labels1.sum().item() and labels1.sum().item() == outputs1.sum().item() :
            result["total_acc"] += 1

    outputs = torch.stack(outputs_res, 0)
    labels = torch.stack(labels_res, 0)
    for i in range(nr_classes) :
        outputs1 = (outputs[:, i].float() >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += (labels1 * outputs1).sum().item() + ((1 - labels1) * (1 - outputs1)).sum().item()
        result["class"][i]["TP"] += (labels1 * outputs1).sum().item()
        result["class"][i]["FN"] += (labels1 * (1 - outputs1)).sum().item()
        result["class"][i]["FP"] += ((1 - labels1) * outputs1).sum().item()
        result["class"][i]["TN"] += ((1 - labels1) * (1 - outputs1)).sum().item()
    return result

accuracy_function_dic = {
    "SingleLabelTop1" : single_label_top1_accuracy,
    "MultiLabel" : multi_label_accuracy,
}

def init_accuracy_function(config, *args, **params):
    name = config.get("output", "accuracy_method")
    if name in accuracy_function_dic :
        return accuracy_function_dic[name]
    else :
        raise NotImplementedError