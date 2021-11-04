from __future__ import absolute_import, division, print_function

import logging
import os
import random
import time

import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import transformers 
from transformers import BertConfig

from opt import get_args
from loader import DataLoader
from model import (SyntaxBertModel, set_seed)
from utils.metrics import correlation_distance, correlation_depth
from utils.constant import pretrained_bert_urls

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)

def evaluate(dataloader,model,probe_type,predict_only=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    syntactic_metric_per_length = defaultdict(list)
    model.eval()

    if probe_type == "distance":
        probe_func = correlation_distance
        gold_ix = 3
    elif probe_type == "depth":
        probe_func = correlation_depth
        gold_ix = 4

    for batch in dataloader:
        with torch.no_grad():
            loss, logits = model(**batch)[:2]
            if not predict_only:
               preds = logits.detach().cpu().numpy()
               eval_loss += loss.item()
               nb_eval_steps += 1
            probe_func(batch[gold_ix],preds,syntactic_metric_per_length)
    
    mean_correlations_per_length = {length:np.mean(syntactic_metric_per_length[length]) for length in syntactic_metric_per_length}
    eval_score = np.mean([mean_correlations_per_length[length] for length in mean_correlations_per_length if 5 <= length <= 50])
    
    if not predict_only:
        eval_loss = eval_loss / nb_eval_steps
        return eval_loss, eval_score
    else:
        return eval_score

def main():
    start_time = time.time()
    args = get_args()
    
    # Setup CUDA, GPU & distributed training
    if not args.force_cpu and not torch.cuda.is_available():
        logger.info("NO available GPU. STOPPED. If you want to continue without GPU, add --force_cpu")
        return 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename="inference_log",filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    if not args.config_name_or_path:
        config_file_name = f"./config/{args.model_type}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    config = BertConfig.from_pretrained(args.config_name_or_path)

    test_dataloader = DataLoader(args.data_dir,"test",args.mode,args.seed,args.batch_size,args.device)

    set_seed(args.seed)
    input_model_dir = os.path.join(args.model_dir,f"{args.model_type}_{args.probe_type}_probe_{args.layer_index}")
    model = SyntaxBertModel.from_pretrained(input_model_dir,config=config,mode=args.mode,
                                            layer_index=args.layer_index,probe_type=args.probe_type)
    model.to(args.device)

    test_score = evaluate(test_dataloader,model,args.probe_type,True)
    with open("./probe_results.txt","a+") as f:
        f.write(f"{args.model_type}\t{args.probe_type}\t{args.layer_index}\t{test_score}\n")
    
    end_time = time.time()
    logger.info(f"time consumed (inference): {(end_time-start_time):.3f} s.")
    logger.info("probe score on the test set saved.")
    
if __name__ == "__main__":
    main()
