import os
import torch
import logging
logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_gpu(gpu : str) :
    gpu_list = []
    if gpu is None :
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else :
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device_list = gpu.split(",")
        for number in range(len(device_list)) :
            gpu_list.append(int(number))
    
    cuda = torch.cuda.is_available()
    logger.info("CUDA available: {}".format(str(cuda)))
    if not cuda and len(gpu_list) :
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError