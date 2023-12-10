from __future__ import absolute_import, division, print_function

import logging
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import (BertConfig, get_linear_schedule_with_warmup)
from model import SyntaxBert, MTSBert
from sklearn.metrics import f1_score
from opt import get_args
from loader import DataLoader

logger = logging.getLogger(__name__)

def one_hot(vector,num_labels):
    res = np.zeros((len(vector),num_labels),dtype=np.int)
    for i, v in enumerate(vector): 
        res[i,v] = 1
    return res

def evaluate(dataloader,model,num_labels,predict_only=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    full_preds = []
    full_golds = []
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            if len(outputs) == 1:
                logits = outputs[0]
            else:
                loss, logits = outputs[:2]
            preds = logits.detach().cpu().numpy()
            if not predict_only:
               eval_loss += loss.item()
               nb_eval_steps += 1
               full_golds.append(batch["labels"].detach().cpu().numpy())
            full_preds.append(np.argmax(preds,axis=1))
    full_preds = np.concatenate(full_preds)
    if not predict_only:
        eval_loss = eval_loss / nb_eval_steps
        full_golds = np.concatenate(full_golds)
        return eval_loss, f1_score(full_golds,one_hot(full_preds,num_labels),average="micro",labels=list(range(1,num_labels))), full_preds
    else:
        return full_preds    
        
def train(args,train_dataloader,dev_dataloader,model,output_dir):
    """ Train the model """
    n_params = sum([p.nelement() for p in model.parameters()])
    print(f'* number of parameters: {n_params}')

    optimizer = AdamW(model.parameters(),lr=args.learning_rate)
    logger.info(f"learning rate: {args.learning_rate}")

    t_total = len(train_dataloader) * args.num_epoches
    if args.warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_ratio*t_total,num_training_steps=t_total)
        logger.info(f"use warmup: {int(args.warmup_ratio*100)} %  steps for warmup.")
    logger.info(f"number of epochs:{args.num_epoches}; number of steps:{t_total}")
    
    best_checkpoint_dir = os.path.join(output_dir,"ckpt")
    if not os.path.exists(best_checkpoint_dir):
        os.makedirs(best_checkpoint_dir)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_epoches)
    logger.info("  Instantaneous batch size = %d", args.batch_size)

    global_step = 0
    logging_loss = 0.0
    max_score = -np.inf
    model.zero_grad()

    for epoch in range(args.num_epoches):
        tr_loss = 0.0
        logging_loss = 0.0
        
        for step, batch in enumerate(train_dataloader): 
            model.train()
            loss = model(**batch)[0] 

            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)              
            tr_loss += loss.item()

            optimizer.step()
            if args.warmup:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                # Log metrics
                logger.info(f"training loss = {(tr_loss - logging_loss)/args.logging_steps} | global step = {global_step}")
                logging_loss = tr_loss

        dev_loss, dev_score, dev_preds = evaluate(dev_dataloader,model,args.num_labels)
        if args.warmup:
            logger.info(f"current lr = {scheduler.get_lr()[0]}")
        logger.info(f"validation loss = {dev_loss} | validation F1-score = {dev_score} | epoch = {epoch}")
        
        if dev_score > max_score:
            max_score = dev_score
            model.save_pretrained(best_checkpoint_dir)
            torch.save(args,os.path.join(best_checkpoint_dir,"training_args.bin"))
            logger.info(f"new best checkpoint! saved under {best_checkpoint_dir}.")
            
            with open(os.path.join(output_dir,f"dev_preds_{args.run_id}.npy"), "wb") as fp:
                np.save(fp,dev_preds)
            logger.info("prediction on the validation set saved.")

    return best_checkpoint_dir

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def main():
    start_time = time.time()

    args = get_args()
    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if not os.path.exists("logging"):
        os.makedirs("logging")
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts")

    config = BertConfig.from_pretrained(args.config_path,num_labels=args.num_labels)
   
    if args.dry_run:
        args.num_epoches = 2    
 
    # Set formats of the output directory
    if args.model_type in ["no_syntax","ct"]:
        dir_suffix = f"{args.dataset_name}_{args.model_type}_{args.learning_rate}"
    elif args.model_type in ["late_fusion","ce","extra"]:
        dir_suffix = f"{args.dataset_name}_{args.model_type}_{args.learning_rate}_{args.num_extra_attention_layers}"
    elif args.model_type == "mts":
        dir_suffix = f"{args.dataset_name}_{args.model_type}_{args.learning_rate}_{args.alpha}"
    else:
        raise ValueError("invalid model type. Valid options (6): no_syntax; extra; ce; ct; late_fusion; mts.")
    output_dir = os.path.join(args.output_dir,dir_suffix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging
    log_fn = os.path.join("./logging/",f"{dir_suffix}_{args.run_id}")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=log_fn,filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    train_dataloader = DataLoader(args,"train")
    dev_dataloader = DataLoader(args,"dev")
    if not args.no_test:
        test_dataloader = DataLoader(args,"test",inference=True)
        logging.info("test set loaded.")
    else:
        logging.info("hyperparameter search: test set not loaded.")

    # training     
    torch.cuda.empty_cache()
    set_seed(args)
    logger.info(f"start training...seed:{args.seed}")
    
    if args.model_type == "mts":
        model = MTSBert.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                        config=config,
                                        args=args)
    else:
        model = SyntaxBert.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                           config=config,
                                           args=args)
    model.to(args.device)

    best_ckpt = train(args,train_dataloader,dev_dataloader,model,output_dir)
       
    if not args.no_test:
        # load the best checkpoint from training
        if args.model_type == "mts":
            model = MTSBert.from_pretrained(best_ckpt,config=config,args=args)
        else:
            model = SyntaxBert.from_pretrained(best_ckpt,config=config,args=args)
        model.to(args.device)
  
        test_preds = evaluate(test_dataloader,model,args.num_labels,predict_only=True)
        with open(os.path.join(output_dir,f"test_preds_{args.run_id}.npy"), "wb") as fp:
            np.save(fp,test_preds)
        logger.info("prediction on the test set saved.")

    end_time = time.time()
    logging.info(f"finished in {(end_time-start_time)/60} minutes.")
    
if __name__ == "__main__":
    main()
