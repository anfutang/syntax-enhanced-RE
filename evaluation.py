from __future__ import absolute_import, division, print_function

import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle

import torch
import transformers 
from transformers import BertConfig
from sklearn.metrics import f1_score

from opt import get_args
from loader import DataLoader
from model import (SyntaxBertModel, set_seed)
from utils.metrics import correlation_distance, correlation_depth
from utils.constant import pretrained_bert_urls

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)

def oh(v):
    oh_vectors = np.zeros_like(v)
    for i, line in enumerate(v):
        oh_vectors[i,np.argmax(line)] = 1
    return oh_vectors

def evaluate(dataloader,model,mode,probe_type=None,predict_only=False,return_prediction=False):
    #print("evaluating...")
    eval_loss = 0.0
    nb_eval_steps = 0
    syntactic_metric_per_length = defaultdict(list)
    model.eval()

    if mode == "probe_only":
        if probe_type == "distance":
            probe_func = correlation_distance
            gold_attrib = "dist_matrixs"
        elif probe_type == "depth":
            probe_func = correlation_depth
            gold_attrib = "depths"
    elif mode == "no_syntax":
        gold_attrib = "labels"
        all_preds = []
        all_golds = []
    
    if return_prediction:
        all_preds = []

    for batch in dataloader:
        with torch.no_grad():
            if predict_only:
                logits = model(**batch,predict_only=predict_only)[0]
            else:
                loss, logits = model(**batch,predict_only=predict_only)[:2]
            
            if mode == "probe_only":
                if probe_type == "distance":
                    preds = [((t.unsqueeze(1) - t.unsqueeze(0))**2).sum(-1) for t in logits]
                elif probe_type == "depth":
                    preds = [(t**2).sum(-1) for t in logits]
                preds = [t.detach().cpu().numpy() for t in preds] 
                golds = [t.detach().cpu().numpy() for t in batch[gold_attrib]]
                masks = [t.eq(0).detach().cpu().numpy() for t in batch["masks"]]
                probe_func(golds,preds,masks,syntactic_metric_per_length) 
            else:
                all_preds.append(oh(logits.detach().cpu().numpy()))
                all_golds.append(batch[gold_attrib].detach().cpu().numpy())
            
            if return_prediction and mode == "probe_only":
                all_preds += preds
            if not predict_only:
                eval_loss += loss.item()
                nb_eval_steps += 1
    
    if mode == "probe_only":
        mean_correlations_per_length = {length:np.mean(syntactic_metric_per_length[length]) for length in syntactic_metric_per_length}
        eval_score = np.mean([mean_correlations_per_length[length] for length in mean_correlations_per_length if 5 <= length <= 50])
    else:
        all_golds = np.concatenate(all_golds)
        all_preds = np.concatenate(all_preds)
        #print(all_golds.shape,all_preds.shape)
        #print(all_golds)
        #print('*'*10)
        #print(all_preds)
        #print('*'*10)
        eval_score = f1_score(all_golds,all_preds,average="micro",labels=[1,2,3,4,5])
           
    if not predict_only:
        eval_loss = eval_loss / nb_eval_steps
        return eval_loss, eval_score
    else:
        eva_outputs = (eval_score,)
        if return_prediction:
            eva_outputs += ([list(l) for l in all_preds],)
        return eva_outputs

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
    log_fn = "logging/inference_log"
    if args.probe_only_no_train:
        log_fn = "logging/inference_log_no_trained_probe"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=log_fn,filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    if not args.config_name_or_path:
        config_file_name = f"syntax-enhanced-RE/config/{args.model_type}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    config = BertConfig.from_pretrained(args.config_name_or_path)

    test_dataloader = DataLoader(args.data_dir,"test",args.mode,args.seed,args.batch_size,args.device)

    set_seed(args)
    if args.mode == "probe_only":
        #input_model_dir = os.path.join(args.model_dir,f"{args.mode}_{args.model_type}_{args.probe_type}_probe_{args.layer_index}")
        input_model_dir = os.path.join(args.model_dir,f"{args.model_type}_{args.probe_type}_probe_{args.layer_index}")
    else:
        input_model_dir = os.path.join(args.model_dir,f"finetune_{args.mode}_{args.model_type}_seed_{args.seed}_ensemble_{args.ensemble_id}")    

    train_probe = True
    if args.probe_only_no_train:
        assert args.mode == "probe_only", "set the option probe_only_no_train ONLY WHEN the mode is PROBE_ONLY"
        train_probe = False
        input_model_dir = pretrained_bert_urls[args.model_type]

    model = SyntaxBertModel.from_pretrained(input_model_dir,config=config,mode=args.mode,dataset_name=args.dataset_name,num_labels=args.num_labels,
                                            layer_index=args.layer_index,probe_type=args.probe_type,train_probe=train_probe)
    model.to(args.device)

    eva_outputs = evaluate(test_dataloader,model,args.mode,args.probe_type,True,args.save_predictions)
    if args.probe_only_no_train:
        output_fn = "./no_trained_probe_results.txt"
    else:
        output_fn = "./probe_results.txt"

    with open(output_fn,"a+") as f:
        f.write(f"{args.model_type}\t{args.mode}\t{args.probe_type}\t{args.layer_index}\t{args.probe_rank}\t{args.ensemble_id}\t{eva_outputs[0]}\n")

    if args.save_predictions:
        with open(os.path.join(input_model_dir,"preds.pkl"),"wb") as f:
            pickle.dump(eva_outputs[1],f,pickle.HIGHEST_PROTOCOL)
            logger.info("probe predictions saved.")
    
    end_time = time.time()
    logger.info(f"time consumed (inference): {(end_time-start_time):.3f} s.")
    logger.info("probe score on the test set saved.")
    
if __name__ == "__main__":
    main()
