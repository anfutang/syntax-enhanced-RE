import os
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').disabled = True

import numpy as np
import pickle
import nltk
import benepar
import time
from tqdm import tqdm
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

const_parser = benepar.Parser("benepar_en3")

def dig(tree):
    # there exists exceptional cases like  S - VP - NP - (subtree with more than one branch)
    # need to distinguish if a subtree of length 1 is: 1) a leave; 2) a subtree with more than one branch.
    for t in tree:
        res = type(t) is str
        if type(t) != str:
            if len(t) > 1:
                return False
            res = dig(t)
    # if subtree is a string: it's a leave;
    # elif subtree is not string and length > 1: it's a subtree;
    # else: continue digging until one of the two conditions above holds;
    return res

def dfs(parse_tree,ix):
    # return: (number of leaves in the subtree, lists of chunk indexes);
    # dfs the constituency parse tree: pick leaves at the same level and group them;
    s = 0
    seq = []
    tmp_seq = []
    for tree in parse_tree:
        if len(tree) == 1 and dig(tree):
            tmp_seq.append(ix)
            s += 1
            ix += 1
        else:
            if tmp_seq:
                seq.append(tmp_seq)
                tmp_seq = []
            ss, sub_seq = dfs(tree,ix)
            ix += ss
            s += ss
            seq += sub_seq
    if tmp_seq:
        seq.append(tmp_seq)
    return s, seq

"""def linearise_parse_tree(parse_tree,ix):
    # insert contituency tags of subtrees with more than one leave into original sentences;
    # in this way get the linearised constituency trees.
    s = 0
    if parse_tree.label() != "TOP":
        seq = ['(',f"[{parse_tree.label()}]"]
    else:
        seq = []
    for tree in parse_tree:
        if len(tree) == 1 and dig(tree):
            seq.append(ix)
            s += 1
            ix += 1
        else:
            ss, sub_seq = linearise_parse_tree(tree,ix)
            ix += ss
            s += ss
            seq += sub_seq
    if parse_tree.label() != "TOP":
        seq += [')']
    return s, seq

def check(cleaned_words,const_spans,seq):
    max_index = max(map(max,const_spans))
    assert max_index == len(cleaned_words) - 1, "word indexes missing in constituency spans."
    max_index = -1
    num_parentheses = 0
    for c in seq:
        if c == '(':
            num_parentheses += 1
        elif c == ')':
            num_parentheses -= 1
        elif type(c) == int:
            max_index = max(max_index,c)
    return num_parentheses == 0 and max_index == len(cleaned_words) - 1"""

def linearise_parse_tree(parse_tree,ix):
    # insert contituency tags of subtrees with more than one leave into original sentences;
    # in this way get the linearised constituency trees.
    s = 0
    if parse_tree.label() != "TOP":
        seq = [f"[{parse_tree.label()}]"]
    else:
        seq = []
    for tree in parse_tree:
        if len(tree) == 1 and dig(tree):
            seq.append(ix)
            s += 1
            ix += 1
        else:
            ss, sub_seq = linearise_parse_tree(tree,ix)
            ix += ss
            s += ss
            seq += sub_seq
    return s, seq

def check(cleaned_words,const_spans,seq):
    max_index = max(map(max,const_spans))
    assert max_index == len(cleaned_words) - 1, "word indexes missing in constituency spans."
    max_index = -1
    for c in seq:
        if type(c) == int:
            max_index = max(max_index,c)
    return max_index == len(cleaned_words) - 1

def parse(cleaned_words,parser):
    input_sentence = benepar.InputSentence(words=cleaned_words)
    try:
        p = parser.parse(input_sentence)
    except:
        return False, [], []
    s, const_spans = dfs(p,0)
    assert s == len(cleaned_words), "words missing when calculating constituency spans."
    s, const_seq = linearise_parse_tree(p,0)
    assert s == len(cleaned_words), "words missing when linearizing constituency trees."
    return check(cleaned_words,const_spans,const_seq), const_spans, const_seq

if __name__ == "__main__":
    parser = ArgumentParser(description='Data Preparation for Syntax-enhanced RE models: remove entity markers from sentences.')
    parser.add_argument("--data_fn", default=None, type=str,help="path to word-level files. NOTE: should be an accessible filename (.pkl file).")
    parser.add_argument("--dataset_name",type=str,help="train, dev or test. Can also be other unique names.")
    parser.add_argument("--output_dir",default=None,type=str,help="directory to output. NOTE: should be an accessible directory.")
    args = parser.parse_args()

    data = pickle.load(open(args.data_fn,"rb"))
    
    start_time = time.time()

    # since benepar has a limitation of 512 on the number of words, truncate sequences of length > 512.
    error_indexes = []
    spans, seqs = [], []
    for ix in tqdm(range(len(data))):
        cleaned_words = data[ix]["cleaned_words"]
        try:
            t, const_spans, const_seq = parse(cleaned_words,const_parser)
            if not t:
                error_indexes.append(ix)
            spans.append(const_spans)
            seqs.append(const_seq)
        except:
            print(ix)
            error_indexes.append(ix)
            spans.append([])
            seqs.append([])
        
    if len(error_indexes) > 0:
        pickle.dump(error_indexes,open(os.path.join(args.output_dir,f"{args.dataset_name}_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
        print(f"{len(error_indexes)} errors.")
    else:
        print("No Error.")
    pickle.dump(spans,open(os.path.join(args.output_dir,f"{args.dataset_name}_const_spans.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump(seqs,open(os.path.join(args.output_dir,f"{args.dataset_name}_const_seqs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    end_time = time.time()
    print(f"time: {end_time-start_time} s.")
    print("finished.")