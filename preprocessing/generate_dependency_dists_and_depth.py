import os
import numpy as np
import pickle
from tqdm import tqdm, trange
from argparse import ArgumentParser
from collections import deque, defaultdict

def get_dists(graph,start,N):
    dist = [0] * N
    visited = set([start])
    q = deque([(start,0)])
    while q:
        curr_node, d = q.popleft()
        dist[curr_node] = d
        for node in graph[curr_node]:
            if node not in visited:
                q.append((node,d+1))
                visited.add(node)
    return dist

def get_pairwise_distances_and_depth(deps):
    # dists: [[0-0,0-1,...,0-(N-1)],[1-1,1-2,...,1-(N-1)],...[(N-2)-(N-1)]]
    g = defaultdict(list)
    for i,j,dep in deps:
        if dep == "root":
            root_ix = i
        else:
            g[i].append(j)
            g[j].append(i)
    N = len(deps)
    dists = []
    for ix in range(N):
        tmp_dist = get_dists(g,ix,N)
        if ix == root_ix:
            depths = tmp_dist
        dists.append(tmp_dist[ix:])
    return dists, depths

def extract_only_word_spans(wps,wp_spans,cleaned_words,subj_marker_indexes,obj_marker_indexes):
    new_wp_spans = []
    for ix, span in enumerate(wp_spans):
        if ix not in [0,len(wp_spans)-1] + subj_marker_indexes + obj_marker_indexes:
            new_wp_spans.append(span)
    cws = []
    for i,j in new_wp_spans:
        tmp_wps = []
        for wp in wps[i:j]:
            if wp.startswith("##"):
                tmp_wps.append(wp[2:])
            else:
                tmp_wps.append(wp)
        cws.append(''.join(tmp_wps))
    return cws == cleaned_words, new_wp_spans

def filter(spans):
    N = len(spans)
    for i in range(N-1,-1,-1):
        if spans[i][1] <= 512:
            break
    return spans[:i+1]

def process(wp_fn,word_fn,dep_fn,desc):
    cleaned_wp_spans = []; dists = []; depths = []
    error_indexes = []
    for ix in trange(len(word_fn),desc=desc):
        wps = wp_fn["wps"][ix]
        wp_spans = wp_fn["spans"][ix]
        item = word_fn[ix]
        cleaned_words = item["cleaned_words"]
        if item["num_entities"] == 2:
            subj_marker_indexes = item["subj_marker_indexes"]
            obj_marker_indexes = item["obj_marker_indexes"]
        else:
            subj_marker_indexes = item["subj_obj_marker_indexes"]
            obj_marker_indexes = []
        deps = dep_fn[ix]
        f, tmp_spans = extract_only_word_spans(wps,wp_spans,cleaned_words,subj_marker_indexes,obj_marker_indexes)
        dist, depth = get_pairwise_distances_and_depth(deps)
        f = f and len(dist) == len(deps) and len(depth) == len(deps)
        if not f:
            error_indexes.append(ix)
        if len(wps) > 512:
            tmp_spans = filter(tmp_spans)
            filtered_N = len(tmp_spans)
            dist = [d[:filtered_N-ii] for ii, d in enumerate(dist[:filtered_N])]
            depth = depth[:filtered_N]
        cleaned_wp_spans.append(tmp_spans)
        dists.append(dist)
        depths.append(depth)
    return cleaned_wp_spans, dists, depths, error_indexes


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
    train_spans, train_dists, train_depths, train_error_indexes = process(train_wp_fn,train_word_fn,train_dep_fn,"train")
    train_data = {"spans":train_spans,"distances":train_dists,"depths":train_depths}
    pickle.dump(train_data,open(os.path.join(args.output_dir,"train_probe.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    if len(train_error_indexes) != 0:
        print(f"train:{len(train_error_indexes)} errors.")
        pickle.dump(train_error_indexes,open(os.path.join(args.output_dir,"train_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    # dev
    dev_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"dev.pkl"),"rb"))
    dev_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"dev.pkl"),"rb"))
    dev_dep_fn = pickle.load(open(os.path.join(args.dep_parse_dir,"dev.pkl"),"rb"))
    dev_spans, dev_dists, dev_depths, dev_error_indexes = process(dev_wp_fn,dev_word_fn,dev_dep_fn,"dev")
    dev_data = {"spans":dev_spans,"distances":dev_dists,"depths":dev_depths}
    pickle.dump(dev_data,open(os.path.join(args.output_dir,"dev_probe.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    if len(dev_error_indexes) != 0:
        print(f"dev:{len(dev_error_indexes)} errors.")
        pickle.dump(dev_error_indexes,open(os.path.join(args.output_dir,"dev_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    # test
    test_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"test.pkl"),"rb"))
    test_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"test.pkl"),"rb"))
    test_dep_fn = pickle.load(open(os.path.join(args.dep_parse_dir,"test.pkl"),"rb"))
    test_spans, test_dists, test_depths, test_error_indexes = process(test_wp_fn,test_word_fn,test_dep_fn,"test")
    test_data = {"spans":test_spans,"distances":test_dists,"depths":test_depths}
    pickle.dump(test_data,open(os.path.join(args.output_dir,"test_probe.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    if len(test_error_indexes) != 0:
        print(f"test:{len(test_error_indexes)} errors.")
        pickle.dump(test_error_indexes,open(os.path.join(args.output_dir,"test_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    
    print("finished.")
