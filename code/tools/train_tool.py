import logging
logger = logging.getLogger(__name__)
import os
import torch
from timeit import default_timer as timer

from tools.eval_tool import valid, gen_time_str, output_value
import transformers

def checkpoint(filename, model, optimizer, trained_epoch, config, global_step) :
    model_to_save = model.module if hasattr(model, "module") else model
    save_params = {
        "model" : model_to_save.state_dict(),
        "optimizer_name" : config.get("train", "optimizer"),
        "optimizer" : optimizer.state_dict(),
        "trained_epoch" : trained_epoch,
        "global_step" : global_step
    }
    try :
        torch.save(save_params, filename)
    except Exception as e :
        logger.warning("Cannot save models with error {}, continue anyway".format(str(e)))

def check(parameters, best_f1, model, current_epoch, optimizer, output_path, config, output_function, global_step) :
    cur_f1, info = valid(parameters, model, parameters["valid_dataset"], current_epoch, config, output_function, mode = "valid")
    if cur_f1 > best_f1 :
        if parameters["tuned_model"] is not None :
            checkpoint(os.path.join(output_path, "ckpt.bin"), parameters["tuned_model"], optimizer, current_epoch, config, global_step)
        best_f1 = cur_f1
        info = valid(parameters, model, parameters["test_dataset"], current_epoch, config, output_function, mode = "test")[1]
        with open(os.path.join(output_path, "test_result.out"), "w") as f :
            f.write(str(info))
    return best_f1

def train(parameters, config) :
    try :
        step_size = config.getint("eval", "step_size")
    except :
        step_size = None

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path) :
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok = True)

    model = parameters["model"]
    optimizer = parameters["optimizer"]
    global_step = 0
    output_function = parameters["output_function"]
    
    if parameters["run_mode"] == "test" :
        check(parameters, 0.0, model, 0, optimizer, output_path, config, output_function, global_step)
        return

    tuned_model = parameters["tuned_model"]
    dataset = parameters["train_dataset"]
    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")
    epoch = config.getint("train", "epoch")
    warmup_proportion = 0.1
    num_train_steps = epoch * len(dataset)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(warmup_proportion * num_train_steps), num_training_steps = num_train_steps)
    
    total_len = len(dataset)
    best_f1 = 0.0
    logger.info("Training start....")
    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
    for current_epoch in range(1, epoch + 1):
        start_time = timer()
        acc_result = None
        total_loss = 0
        for step, data in enumerate(dataset) :
            if config.get("model", "model_name").startswith("FewRel") :
                data = parameters["Formatter"].process("train", data)
            tuned_model.train()
            model.train()
            optimizer.zero_grad()
            results = model(data, config, acc_result, parameters["embedding"], parameters["NeedMapper"], parameters["mapper"], "train")
            loss, acc_result = results["loss"], results["acc_result"]
            if type(loss) != float :
                total_loss += loss.item()
                loss.backward()
            else :
                total_loss += loss
            optimizer.step()
            scheduler.step()
            if step % output_time == 0 :
                output_info = output_function(acc_result, config)
                delta_t = timer() - start_time
                output_value(current_epoch, "train", "{}/{}".format(step + 1, total_len), "{}/{}".format(
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            round(total_loss / (step + 1), 4), output_info, "\r", config)
            if step_size is not None and global_step % step_size == 0 :
                best_f1 = check(parameters, best_f1, model, current_epoch, optimizer, output_path, config, output_function, global_step)
            global_step += 1
        if current_epoch % test_time == 0 :
            best_f1 = check(parameters, best_f1, model, current_epoch, optimizer, output_path, config, output_function, global_step)