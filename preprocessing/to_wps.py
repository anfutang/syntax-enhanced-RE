import os
import re
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer

pretrained_bert_urls = {"bert":"bert-base-uncased",
                        "biobert":"dmis-lab/biobert-base-cased-v1.1",
                        "scibert":"allenai/scibert_scivocab_uncased",
                        "roberta":"roberta-base"}

def get_wps_to_tokens(tokens,wp_ids,tokenizer):
    wp_to_token_map = [1]
    curr_id = 1
    wps = ["[CLS]"]
    for token in tokens:
        tmp_wps = tokenizer.tokenize([token],is_split_into_words=True)
        curr_id += len(tmp_wps)
        wps += tmp_wps
        wp_to_token_map.append(curr_id)
    wps += ["[SEP]"]
    wp_to_token_map.append(curr_id+1)
    restore_tokens = restore_tokens_from_wps(wps,wp_to_token_map)
    #print(restore_tokens)
    correct = token_wise_compare(restore_tokens[1:-1],[token.replace(' ','') for token in tokens])
    correct = correct and (tokenizer.convert_tokens_to_ids(wps)==wp_ids)
    return wp_to_token_map, correct

def token_wise_compare(lst_1,lst_2):
    ans = True
    for token_1, token_2 in zip(lst_1, lst_2):
        if "[UNK]" in token_1:
            continue
        ans = ans and (token_1 == token_2)
    return ans

def restore_tokens_from_wps(wps,wp_ids):
    curr_i = 0
    restored_tokens = []
    for i in wp_ids:
        curr_token = []
        for wp in wps[curr_i:i]:
            if wp[:2] == "##":
                curr_token.append(wp[2:])
            else:
                curr_token.append(wp)
            curr_i = i
        restored_tokens.append(''.join(curr_token))
    return restored_tokens

def main(args):
    DIR = args.data_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_urls[args.bert_type],do_lower_case=False)

    train = pickle.load(open(os.path.join(DIR,"train.pkl"),"rb"))
    dev = pickle.load(open(os.path.join(DIR,"dev.pkl"),"rb"))
    test = pickle.load(open(os.path.join(DIR,"test.pkl"),"rb"))

    wp_ids_train = tokenizer([d["tokens"] for d in train],is_split_into_words=True,add_special_tokens=True)["input_ids"]
    wp_ids_dev = tokenizer([d["tokens"] for d in dev],is_split_into_words=True,add_special_tokens=True)["input_ids"]
    wp_ids_test= tokenizer([d["tokens"] for d in test],is_split_into_words=True,add_special_tokens=True)["input_ids"]

    print(f"maximum length of wordpiece sequences:\ntrain:{max(map(len,wp_ids_train))}\ndev:{max(map(len,wp_ids_dev))}\n"  
                                                    f"test:{max(map(len,wp_ids_test))}")

    wp_to_token_map_train = []
    correct_on_train = True
    for i, d in tqdm(enumerate(train),desc="train"):
        tmp_map, c = get_wps_to_tokens(d["tokens"],wp_ids_train[i],tokenizer)
        wp_to_token_map_train.append(tmp_map)
        correct_on_train = correct_on_train and c
    
    assert correct_on_train, "TRAIN: wordpiece to token map not correct somewhere."
    assert len(wp_to_token_map_train) == len(wp_ids_train), "TRAIN: length of wordpiece sequences and maps NOT equal."

    with open(os.path.join(args.output_dir,"wp_train.pkl"),"wb") as f:
        pickle.dump({"wps":wp_ids_train,"map":wp_to_token_map_train},f,pickle.HIGHEST_PROTOCOL)
    
    wp_to_token_map_dev = []
    correct_on_dev = True
    for i, d in tqdm(enumerate(dev),desc="dev"):
        tmp_map, c = get_wps_to_tokens(d["tokens"],wp_ids_dev[i],tokenizer)
        wp_to_token_map_dev.append(tmp_map)
        correct_on_dev = correct_on_dev and c
    
    assert correct_on_dev, "DEV: wordpiece to token map not correct somewhere."
    assert len(wp_to_token_map_dev) == len(wp_ids_dev), "DEV: length of wordpiece sequences and maps NOT equal."

    with open(os.path.join(args.output_dir,"wp_dev.pkl"),"wb") as f:
        pickle.dump({"wps":wp_ids_dev,"map":wp_to_token_map_dev},f,pickle.HIGHEST_PROTOCOL)

    wp_to_token_map_test = []
    correct_on_test = True
    for i, d in tqdm(enumerate(test),desc="test"):
        tmp_map, c = get_wps_to_tokens(d["tokens"],wp_ids_test[i],tokenizer)
        wp_to_token_map_test.append(tmp_map)
        correct_on_test = correct_on_test and c
    
    assert correct_on_test, "TEST: wordpiece to token map not correct somewhere."
    assert len(wp_to_token_map_test) == len(wp_ids_test), "TEST: length of wordpiece sequences and maps NOT equal."

    with open(os.path.join(args.output_dir,"wp_test.pkl"),"wb") as f:
        pickle.dump({"wps":wp_ids_test,"map":wp_to_token_map_test},f,pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = ArgumentParser(description='convert token-level data to wordpiece-level data')
     
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                         help="path to token-level data files (results of to_tokens.py)")
    parser.add_argument("--output_dir", default=None, type=str, 
                         help="path to save the outputs.")
    parser.add_argument("--bert_type", default="biobert", type=str)

    args = parser.parse_args()
    main(args)