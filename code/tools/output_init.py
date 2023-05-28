import json
from .accuracy_init import gen_micro_macro_result

def null_output_function(data, config, *args, **params) :
    return ""

def basic_output_function(data, config, *args, **params) :
    which = config.get("output", "output_value").replace(" ", "").split(",")
    if "acc" in which :
        return json.dumps({"acc": round(data['total_acc'] / data['total'], 4)}, sort_keys = True)
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]
    return json.dumps(result, sort_keys = True)

output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function
}

def init_output_function(config, *args, **params) :
    name = config.get("output", "output_function")
    if name in output_function_dic :
        return output_function_dic[name]
    else:
        raise NotImplementedError