import argparse
import torch
from tools.seed import seed_everything
from tools.gpu_tool import set_gpu

from config_parser import create_config
from tools.train_tool import train

import logging
logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from formatter.FewRel_formatter import FewRel_bert_formatter, FewRel_roberta_formatter
from formatter.Wiki80_formatter import Wiki80_bert_formatter
from formatter.WikiET_formatter import WikiET_bert_formatter
from formatter.EntityQuestions_formatter import EntityQuestions_bert_formatter
model_to_formatter = {
                        "FewRel_BERT" : FewRel_bert_formatter, "FewRel_Adapter" : FewRel_bert_formatter, "FewRel_LoRA" : FewRel_bert_formatter, "FewRel_BitFit" : FewRel_bert_formatter,
                        "FewRel_RoBERTa" : FewRel_roberta_formatter,
                        "Wiki80_BERT" : Wiki80_bert_formatter, "Wiki80_Adapter" : Wiki80_bert_formatter, "Wiki80_LoRA" : Wiki80_bert_formatter, "Wiki80_BitFit" : Wiki80_bert_formatter,
                        "WikiET_BERT" : WikiET_bert_formatter, "WikiET_Adapter" : WikiET_bert_formatter, "WikiET_LoRA" : WikiET_bert_formatter, "WikiET_BitFit" : WikiET_bert_formatter,
                        "EntityQuestions_BERT" : EntityQuestions_bert_formatter, "EntityQuestions_Adapter" : EntityQuestions_bert_formatter, "EntityQuestions_LoRA" : EntityQuestions_bert_formatter, "EntityQuestions_BitFit" : EntityQuestions_bert_formatter,
                    }

from model.FewRel_model import FewRel_BERT, FewRel_Adapter, FewRel_LoRA, FewRel_BitFit, FewRel_RoBERTa
from model.Wiki80_model import Wiki80_BERT, Wiki80_Adapter, Wiki80_LoRA, Wiki80_BitFit
from model.WikiET_model import WikiET_BERT, WikiET_Adapter, WikiET_LoRA, WikiET_BitFit
from model.EntityQuestions_model import EntityQuestions_BERT, EntityQuestions_Adapter, EntityQuestions_LoRA, EntityQuestions_BitFit
model_to_model = {
                        "FewRel_BERT" : FewRel_BERT, "FewRel_Adapter" : FewRel_Adapter, "FewRel_LoRA" : FewRel_LoRA, "FewRel_BitFit" : FewRel_BitFit,
                        "FewRel_RoBERTa" : FewRel_RoBERTa,
                        "Wiki80_BERT" : Wiki80_BERT, "Wiki80_Adapter" : Wiki80_Adapter, "Wiki80_LoRA" : Wiki80_LoRA, "Wiki80_BitFit" : Wiki80_BitFit,
                        "WikiET_BERT" : WikiET_BERT, "WikiET_Adapter" : WikiET_Adapter, "WikiET_LoRA" : WikiET_LoRA, "WikiET_BitFit" : WikiET_BitFit,
                        "EntityQuestions_BERT" : EntityQuestions_BERT, "EntityQuestions_Adapter" : EntityQuestions_Adapter, "EntityQuestions_LoRA" : EntityQuestions_LoRA, "EntityQuestions_BitFit" : EntityQuestions_BitFit,
                    }

def init_dataset(config, Formatter, mode : str, start_seed = 0) :
    batch_size = config.getint("train" if mode == "train" else "eval", "batch_size")
    if config.get("model", "model_name").startswith("FewRel") :
        length = config.getint("train" if mode == "train" else "eval", "iteration")
        return [range(index, min(index + batch_size, length + start_seed)) for index in range(0 + start_seed, length + start_seed, batch_size)]
    if mode in ("valid", "test") :
        try :
            batch_size = config.getint("eval", "batch_size")
        except :
            logger.warning("[eval] batch size has not been defined in config file, use [train] batch_size instead.")
    return torch.utils.data.DataLoader(dataset = Formatter.process(mode), batch_size = batch_size, shuffle = True, drop_last = False)

from model.optimizer import init_optimizer
from tools.output_init import init_output_function
def init_all(config, checkpoint, run_mode):
    result = {}
    logger.info("Begin to initialize dataset and formatter...")

    if config.get("model", "model_name").startswith("FewRel") :
        result["Formatter"] = Formatter = model_to_formatter[config.get("model", "model_name")](config)
    else :
        Formatter = model_to_formatter[config.get("model", "model_name")](config)
    
    if run_mode != "test" :
        if config.get("model", "model_name").startswith("FewRel") :
            try :
                start_seed = config.getint("data", "start_seed")
            except :
                start_seed = 0
            result["train_dataset"] = init_dataset(config, Formatter, "train", start_seed)
        else :
            result["train_dataset"] = init_dataset(config, Formatter, "train")
    else :
        result["train_dataset"] = None
    result["valid_dataset"] = init_dataset(config, Formatter, "valid")
    result["test_dataset"] = init_dataset(config, Formatter, "test")
    
    result["embedding"] = Formatter.get_embedding()
    result["NeedMapper"] = Formatter.get_NeedMapper()
    mapper = result["mapper"] = Formatter.get_mapper()

    logger.info("Begin to initialize models...")
    model = model_to_model[config.get("model", "model_name")](config)
    try :
        model.load_state_dict(torch.load(checkpoint, map_location = torch.device("cpu"))["model"])
    except Exception as e :
        logger.warning("Cannot load checkpoint file with error {}".format(str(e)))
    if torch.cuda.is_available() :
        model = model.cuda()
    if mapper is not None :
        mapper.eval()
    model.eval()
    
    if run_mode == "fine-tuning" :
        tuned_model = model
        if mapper is not None :
            mapper.abandon_grad()
    elif run_mode == "map-tuning" :
        tuned_model = mapper
    else : # run_mode == "test"
        tuned_model = None
    result["tuned_model"] = tuned_model
    result["run_mode"] = run_mode
    
    if tuned_model is not None :
        optimizer = init_optimizer(tuned_model, config)
    else :
        optimizer = None
    result["output_function"] = init_output_function(config)
    result["model"] = model
    result["optimizer"] = optimizer
    logger.info("Initialize done.")
    return result

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required = True)
    parser.add_argument("--gpu", "-g", default = None)
    parser.add_argument("--checkpoint", help = "checkpoint file path")
    parser.add_argument("--seed", type = int, default = 233)
    parser.add_argument("--run_mode", type = str, default = "fine-tuning", choices = ("fine-tuning", "map-tuning", "test"))
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    set_gpu(args.gpu)
    seed_everything(args.seed)

    parameters = init_all(config, args.checkpoint, args.run_mode)
    train(parameters, config)