import os
import numpy as np
import pickle
from tqdm import tqdm, trange
from argparse import ArgumentParser


def translate_deps(deps,index_map,subj_marker_indexes,obj_marker_indexes,subj_head,obj_head,num_words):
    for i, j, dep in deps:
        if dep == 'root':
            root_ix = index_map[i]
            break
    new_deps = [(index_map[i],index_map[j],dep) for i,j,dep in deps if j != -1]
    new_deps.append((0,root_ix,"bert_symbol"))
    new_deps.append((num_words-1,root_ix,"bert_symbol"))
    new_deps.append((root_ix,root_ix,"root"))
    for subj_ix in subj_marker_indexes:
        new_deps.append((subj_ix,subj_head,"entity_marker"))
    for obj_ix in obj_marker_indexes:
        new_deps.append((obj_ix,obj_head,"entity_marker"))
    return sorted(new_deps)

def generate_adj_matrix(wps,wp_spans,deps):
    # matrix saved as adjacency lists
    assert len(wp_spans) == len(deps) # no connection for the syntactic root
    adj = {i:[i] for i in range(len(wps))}
    for ix, span in enumerate(wp_spans):
        head = span[0]
        for i in range(head+1,span[1]):
            adj[i].append(head)
            adj[head].append(i)
        if deps[ix][1] != ix:
            c_head = wp_spans[deps[ix][1]][0]
            adj[head].append(c_head)
            adj[c_head].append(head)
    return adj

def filter(adj):
    new_adj = {}
    for i,v in adj.items():
        if i < 512:
            new_adj[i] = [j for j in v if j < 512]
    return new_adj

def process(wp_fn,word_fn,dep_fn,desc):
    N = len(dep_fn)
    adjs = []
    for ix in trange(N,desc=desc):
        wps = wp_fn["wps"][ix]
        wp_spans = wp_fn["spans"][ix]
        item = word_fn[ix]
        words = item["words"]
        cleaned_words = item["cleaned_words"]
        index_map = item["index_map"]
        if item["num_entities"] == 2:
            subj_marker_indexes = item["subj_marker_indexes"]
            obj_marker_indexes = item["obj_marker_indexes"]
            subj_head = item["subj_head"]
            obj_head = item["obj_head"]
        else:
            subj_marker_indexes = item["subj_obj_marker_indexes"]
            obj_marker_indexes = []
            subj_head = item["subj_obj_head"]
            obj_head = -1
        deps = dep_fn[ix]
        new_deps = translate_deps(deps,index_map,subj_marker_indexes,obj_marker_indexes,subj_head,obj_head,len(words))
        adj = generate_adj_matrix(wps,wp_spans,new_deps)
        if len(wps) > 512:
            adj = filter(adj)
        adjs.append(adj)
    return adjs

if __name__ == "__main__":
    parser = ArgumentParser(description='Gererate ready-to-use data for late-fusion syntax-enhanced models.')
    parser.add_argument("--wp_file_dir",type=str,help="directory to wordpiece-level files.")
    parser.add_argument("--word_file_dir",type=str,help="directory to word-level files.")
    parser.add_argument("--dep_parse_dir", default=None,type=str,help="path to dependency parsing files.")
    parser.add_argument("--output_dir",type=str,help="output directory.")
    args = parser.parse_args()

    # read and process train
    train_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"train.pkl"),"rb"))
    train_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"train.pkl"),"rb"))
    train_dep_fn = pickle.load(open(os.path.join(args.dep_parse_dir,"train.pkl"),"rb"))
    train_adjs = process(train_wp_fn,train_word_fn,train_dep_fn,"train")
    pickle.dump(train_adjs,open(os.path.join(args.output_dir,"train_adjs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    # dev
    dev_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"dev.pkl"),"rb"))
    dev_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"dev.pkl"),"rb"))
    dev_dep_fn = pickle.load(open(os.path.join(args.dep_parse_dir,"dev.pkl"),"rb"))
    dev_adjs = process(dev_wp_fn,dev_word_fn,dev_dep_fn,"dev")
    pickle.dump(dev_adjs,open(os.path.join(args.output_dir,"dev_adjs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    # test
    test_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"test.pkl"),"rb"))
    test_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"test.pkl"),"rb"))
    test_dep_fn = pickle.load(open(os.path.join(args.dep_parse_dir,"test.pkl"),"rb"))
    test_adjs = process(test_wp_fn,test_word_fn,test_dep_fn,"test")
    pickle.dump(test_adjs,open(os.path.join(args.output_dir,"test_adjs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    
    print("finished.")