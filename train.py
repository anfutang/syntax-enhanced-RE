from __future__ import absolute_import, division, print_function

import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import transformers 
from transformers import BertConfig
from torch.optim import Adam
#from pure_model import BertForSequenceClassification

from opt import get_args
from loader import DataLoader
from evaluation import evaluate
from model import (SyntaxBertModel, set_seed)
from utils.constant import pretrained_bert_urls

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)

os.environ["MKL_DEBUG_CPU_TYPE"] = '5'

def train(args,train_dataloader,dev_dataloader,model,output_dir):
    """ Train the model """
    if not args.early_stopping:
        NUM_EPOCHS = args.num_train_epochs
    else:
        logger.info(f"early stopping chosen. MAXIMUM number of epochs set to {args.max_num_epochs}.")
        NUM_EPOCHS = args.max_num_epochs

    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    logger.info(f'===number of parameters (pre-trained BERT frozen={args.freeze_bert}): {n_params}')

    t_total = len(train_dataloader) * NUM_EPOCHS
    
    logger.info(f"===ensemble id:{args.ensemble_id}; number of steps:{t_total}; learning rate:{args.learning_rate}")
    optimizer = Adam(model.parameters(),lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataloader)}")
    logger.info(f"Num Epochs = {NUM_EPOCHS}")
    logger.info(f"Batch size = {args.batch_size}")

    global_step = 0
    logging_loss = 0.0
    min_loss, prev_dev_loss = np.inf, np.inf
    max_score, prev_dev_score = -np.inf, -np.inf
    training_hist = []
    model.zero_grad()

    dev_loss_record = []
    dev_score_record = []
    for epoch in range(int(NUM_EPOCHS)):
        tr_loss = 0.0
        logging_loss = 0.0
        grad_norm = 0.0
        #epoch_iterator = tqdm(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            model.train()
            loss, _ = model(**batch)
            loss.backward() # gradient will be stored in the network
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)

            grad_norm += gnorm
                                                
            tr_loss += loss.item()

            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                # Log metrics
                logger.info(f"training loss = {(tr_loss - logging_loss)/args.logging_steps} | global step = {global_step} epoch = {epoch}")
                logging_loss = tr_loss

        dev_loss, dev_score = evaluate(dev_dataloader,model,args.mode,args.probe_type)
        dev_loss_record.append(dev_loss)
        dev_score_record.append(dev_score)

        logger.info(f"validation loss = {dev_loss} | validation F1-score = {dev_score} | epoch = {epoch}")

        if args.monitor == "loss" and dev_loss < min_loss:
            min_loss = dev_loss
            # save model
            model.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("new best model! saved.")
        
        if args.monitor == "score" and dev_score > max_score:
            max_score = dev_score
            # save model
            model.save_pretrained(output_dir)
            torch.save(args,os.path.join(output_dir,"training_args.bin"))
            logger.info("new best model! saved.")
        
        if args.early_stopping and args.monitor == "loss":
            if dev_loss < prev_dev_loss:
                training_hist.append(True)
            else:
                training_hist.append(False)
                if len(training_hist) > args.patience and not np.any(training_hist[-args.patience:]):
                    logger.info(f"early stopping triggered: best loss on validation set: {min_loss}.")
                    break
            prev_dev_loss = dev_loss

        if args.early_stopping and args.monitor == "score":
            if dev_score >= prev_dev_score:
                training_hist.append(True)
            else:
                training_hist.append(False)
                if len(training_hist) > args.patience and not np.any(training_hist[-args.patience:]):
                    logger.info(f"early stopping triggered: best F-score on validation set: {max_score}.")
                    break
            prev_dev_score = dev_score

        if epoch + 1 == NUM_EPOCHS:
            break

    return dev_loss_record, dev_score_record

def main():
    start_time = time.time()
    args = get_args()
   
    logger.info("training...") 
    # Setup CUDA, GPU & distributed training
    if not args.force_cpu and not torch.cuda.is_available():
        logger.info("NO available GPU. STOPPED. If you want to continue without GPU, add --force_cpu")
        return 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if not os.path.exists("./logging/"):
        os.makedirs("./logging/")

    # Setup logging
    if args.mode == "probe_only":
        logging_fn = f"training_log_{args.model_type}_{args.mode}_{args.probe_type}_probe_{args.layer_index}"
    else:
        if args.grid_search:
            logging_fn = f"training_log_{args.model_type}_{args.mode}_gs"
        else:
            logging_fn = f"training_log_{args.model_type}_{args.mode}"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=os.path.join("./logging/",logging_fn),filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    if not args.config_name_or_path:
        config_file_name = f"syntax-enhanced-RE/config/{args.model_type}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    config = BertConfig.from_pretrained(args.config_name_or_path)

    train_dataloader = DataLoader(args.data_dir,"train",args.mode,args.seed,args.batch_size,args.device)
    dev_dataloader = DataLoader(args.data_dir,"dev",args.mode,args.seed,args.batch_size,args.device)
    
    logger.info("data loaded.")
    # Evaluate the best model on Test set
    torch.cuda.empty_cache()
    
    set_seed(args)
    #if args.mode == "no_syntax":
    #    model = BertForSequenceClassification.from_pretrained(pretrained_bert_urls[args.model_type],config=config,num_labels=args.num_labels)
    #else:
    model = SyntaxBertModel.from_pretrained(pretrained_bert_urls[args.model_type],config=config,mode=args.mode,dataset_name=args.dataset_name,num_labels=args.num_labels,
                                            layer_index=args.layer_index,probe_type=args.probe_type,probe_rank=args.probe_rank)
    model.to(args.device)
    if args.freeze_bert:
        #freeze the BERT
        for param in model.bert.parameters():
            param.requires_grad = False

    if args.mode == "probe_only":
        output_model_dir = os.path.join(args.model_dir,f"{args.mode}_{args.model_type}_{args.probe_type}_probe_{args.layer_index}")
    else:
        if args.grid_search:
            output_model_dir = os.path.join(args.model_dir,f"finetune_{args.mode}_{args.model_type}_{args.batch_size}_{args.learning_rate}/seed_{args.seed}_ensemble_{args.ensemble_id}")
        else:
            output_model_dir = os.path.join(args.model_dir,f"finetune_{args.mode}_{args.model_type}_seed_{args.seed}_ensemble_{args.ensemble_id}")

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    loss_record, score_record = train(args,train_dataloader,dev_dataloader,model,output_model_dir)

    df_record = pd.DataFrame({"loss":loss_record,"score":score_record})
    df_record.to_csv(os.path.join(output_model_dir,"training_record.csv"),index=False)
    
    end_time = time.time()
    logger.info(f"time consumed (training): {(end_time-start_time):.3f} s.")
    logger.info("training record saved.")
    
if __name__ == "__main__":
    main()
