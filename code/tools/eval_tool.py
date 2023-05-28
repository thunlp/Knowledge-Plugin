import torch
from timeit import default_timer as timer
from .accuracy_init import gen_micro_macro_result

def gen_time_str(t) :
    t = int(t)
    minute, second = t // 60, t % 60
    return "{}:{}".format(minute, second)

def output_value(epoch, mode, step, time, loss, info, end, config) :
    try :
        delimiter = config.get("output", "delimiter")
    except :
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7 :
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14 :
        s += " "
    s = s + str(step) + " "
    while len(s) < 25 :
        s += " "
    s += str(time)
    while len(s) < 40 :
        s += " "
    s += str(loss)
    while len(s) < 48 :
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if end is not None :
        print(s, end = end)
    else :
        print(s)

def valid(parameters, model, dataset, epoch, config, output_function, mode = "valid") :
    if parameters["tuned_model"] is not None :
        parameters["tuned_model"].eval()
    model.eval()

    acc_result = None
    total_len = len(dataset)
    total_loss = 0.0
    start_time = timer()
    output_time = config.getint("output", "output_time")

    with torch.no_grad() :
        for step, data in enumerate(dataset) :
            if config.get("model", "model_name").startswith("FewRel") :
                data = parameters["Formatter"].process(mode, data)
            results = model(data, config, acc_result, parameters["embedding"], parameters["NeedMapper"], parameters["mapper"], "valid")
            loss, acc_result = results["loss"], results["acc_result"]
            total_loss += float(loss)
            if step % output_time == 0 :
                delta_t = timer() - start_time
                output_info = output_function(acc_result, config)
                output_value(epoch, mode, "{}/{}".format(step + 1, total_len), "{}/{}".format(
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            round(total_loss / (step + 1), 4), output_info, "\r", config)
    return gen_micro_macro_result(acc_result)["micro_f1"], output_info