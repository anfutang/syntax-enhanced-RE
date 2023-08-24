import os
import numpy as np
import pickle
from tqdm import tqdm, trange
from argparse import ArgumentParser

const2id = {'[ADJP]': 28895, '[ADVP]': 28896, '[CONJP]': 28897, '[FRAG]': 28898, '[INTJ]': 28899, '[LST]': 28900, '[NML]': 28901,
            '[NP]': 28902, '[PP]': 28903, '[PRN]': 28904, '[PRT]': 28905, '[QP]': 28906, '[RRC]': 28907, '[SBARQ]': 28908,
            '[SBAR]': 28909, '[SINV]': 28910, '[SQ]': 28911, '[S]': 28912, '[UCP]': 28913, '[VP]': 28914, '[WHADJP]': 28915,
            '[WHADVP]': 28916, '[WHNP]': 28917, '[WHPP]': 28918}


# (1) generate wordpiece spans corresponding to each chunk
def insert_markers(seq,subj_marker_indexes,obj_markers_indexes):
    if not obj_markers_indexes:
        assert len(subj_marker_indexes) == 2, "number of subj-obj markers not eqaul to 2."
        subj_start = subj_marker_indexes[0] + 1
        subj_end = subj_marker_indexes[1] - 1
        subj_obj = True
    elif len(subj_marker_indexes) == 4:
        subj_start = subj_marker_indexes[1] + 1
        subj_end = subj_marker_indexes[2] - 1
        obj_start = obj_markers_indexes[1] + 1
        obj_end = obj_markers_indexes[2] - 1
        subj_obj = False
    else:
        subj_start = subj_marker_indexes[0] + 1
        subj_end = subj_marker_indexes[1] - 1
        obj_start = obj_markers_indexes[0] + 1
        obj_end = obj_markers_indexes[1] - 1
        subj_obj = False
    if subj_obj:
        for i,s in enumerate(seq):
            if subj_start in s:
                ssi = s.index(subj_start)
                seq[i] = s[:ssi] + [subj_marker_indexes[0]] + s[ssi:]
                s = seq[i]
            if subj_end in s:
                sei = s.index(subj_end)
                seq[i] = s[:sei+1] + [subj_marker_indexes[1]] + s[sei+1:]
    elif len(subj_marker_indexes) == 4:
        for i,s in enumerate(seq):
            if subj_start in s:
                ssi = s.index(subj_start)
                seq[i] = s[:ssi] + subj_marker_indexes[:2] + s[ssi:]
                s = seq[i]
            if subj_end in s:
                sei = s.index(subj_end)
                seq[i] = s[:sei+1] + subj_marker_indexes[2:] + s[sei+1:]
                s = seq[i]
            if obj_start in s:
                osi = s.index(obj_start)
                seq[i] = s[:osi] + obj_markers_indexes[:2] + s[osi:]
                s = seq[i]
            if obj_end in s:
                oei = s.index(obj_end)
                seq[i] = s[:oei+1] + obj_markers_indexes[2:] + s[oei+1:]
    else:
        for i,s in enumerate(seq):
            if subj_start in s:
                ssi = s.index(subj_start)
                seq[i] = s[:ssi] + [subj_marker_indexes[0]] + s[ssi:]
                s = seq[i]
            if subj_end in s:
                sei = s.index(subj_end)
                seq[i] = s[:sei+1] + [subj_marker_indexes[1]] + s[sei+1:]
                s = seq[i]
            if obj_start in s:
                osi = s.index(obj_start)
                seq[i] = s[:osi] + [obj_markers_indexes[0]] + s[osi:]
                s = seq[i]
            if obj_end  in s:
                oei = s.index(obj_end)
                seq[i] = s[:oei+1] + [obj_markers_indexes[1]] + s[oei+1:]
    # verify: word indexes corresponding to each constituent must be contiguous
    flag = True
    for s in seq:
        flag = flag and s[-1]-s[0]+1 == len(s)
    return flag, seq
    

def get_wp2const_spans(wp_spans,const_spans,index_map,subj_marker_indexes,obj_marker_indexes):
    wp_spans = {i:span for i, span in enumerate(wp_spans)}
    # constituent spans represented as old word indexes
    const_spans = [[index_map[i] for i in cs] for cs in const_spans]
    f, const_spans = insert_markers(const_spans,subj_marker_indexes,obj_marker_indexes)
    wp2const = []
    for cs in const_spans:
        start_word, end_word = cs[0], cs[-1]
        wp2const.append([wp_spans[start_word][0],wp_spans[end_word][1]])
    return f, [wp_spans[0]] + wp2const + [wp_spans[len(wp_spans)-1]]

# (2) generate linearized constituency trees
def interpret_const_seq(seq,const2id,m,wp_ids,wp_spans,subj_marker_indexes,obj_marker_indexes):
    wp_spans = {i:span for i, span in enumerate(wp_spans)}
    trans_seq = []
    for ch in seq:
        if type(ch) is int:
            trans_seq.append(m[ch])
        else:
            trans_seq.append(ch)
    seq = trans_seq
    if not obj_marker_indexes:
        assert len(subj_marker_indexes) == 2, "number of subj-obj markers not eqaul to 2."
        subj_start = subj_marker_indexes[0] + 1
        subj_end = subj_marker_indexes[1] - 1
        subj_obj = True
    elif len(subj_marker_indexes) == 4:
        subj_start = subj_marker_indexes[1] + 1
        subj_end = subj_marker_indexes[2] - 1
        obj_start = obj_marker_indexes[1] + 1
        obj_end = obj_marker_indexes[2] - 1
        subj_obj = False
    else:
        subj_start = subj_marker_indexes[0] + 1
        subj_end = subj_marker_indexes[1] - 1
        obj_start = obj_marker_indexes[0] + 1
        obj_end = obj_marker_indexes[1] - 1
        subj_obj = False
    if subj_obj:
        if subj_start in seq:
            ssi = seq.index(subj_start)
            seq = seq[:ssi] + [subj_marker_indexes[0]] + seq[ssi:]
        if subj_end in seq:
            sei = seq.index(subj_end)
            seq = seq[:sei+1] + [subj_marker_indexes[1]] + seq[sei+1:]
    elif len(subj_marker_indexes) == 4:
        if subj_start in seq:
            ssi = seq.index(subj_start)
            seq = seq[:ssi] + subj_marker_indexes[:2] + seq[ssi:]
        if subj_end in seq:
            sei = seq.index(subj_end)
            seq = seq[:sei+1] + subj_marker_indexes[2:] + seq[sei+1:]
        if obj_start in seq:
            osi = seq.index(obj_start)
            seq = seq[:osi] + obj_marker_indexes[:2] + seq[osi:]
        if obj_end in seq:
            oei = seq.index(obj_end)
            seq = seq[:oei+1] + obj_marker_indexes[2:] + seq[oei+1:]
    else:
        if subj_start in seq:
            ssi = seq.index(subj_start)
            seq = seq[:ssi] + [subj_marker_indexes[0]] + seq[ssi:]
        if subj_end in seq:
            sei = seq.index(subj_end)
            seq = seq[:sei+1] + [subj_marker_indexes[1]] + seq[sei+1:]
        if obj_start in seq:
            osi = seq.index(obj_start)
            seq = seq[:osi] + [obj_marker_indexes[0]] + seq[osi:]
        if obj_end in seq:
            oei = seq.index(obj_end)
            seq = seq[:oei+1] + [obj_marker_indexes[1]] + seq[oei+1:]
    new_seq = []
    num_wps = 2
    for ch in seq:
        if type(ch) is int:
            tmp_span = wp_spans[ch]
            new_seq += wp_ids[tmp_span[0]:tmp_span[1]]
            num_wps += tmp_span[1] - tmp_span[0]
        else:
            new_seq.append(const2id[ch])
    new_seq = [wp_ids[0]] + new_seq + [wp_ids[-1]]
    return verify_const_seq(new_seq,const2id,wp_ids) and len(wp_ids) == num_wps, new_seq

def verify_const_seq(seq,const2id,wp_ids):
    filtered_seq = []
    for ch in seq:
        if ch not in const2id.values():
            filtered_seq.append(ch)
    return filtered_seq == wp_ids

# process a dataset (train,dev,or test)
def filter_long_results(spans,seq):
    if spans[-1][-1] > 512:
        new_spans = []
        for span in spans:
            if span[1] <= 512:
                new_spans.append(span)
            else:
                if span[0] < 512:
                    new_spans.append([span[0],512])
                    break
    else:
        new_spans = spans 
    if len(seq) > 512:
        seq = seq[:511] + [3]
    return new_spans, seq

def per_example_process(const_spans,seq,wp_ids,wp_spans,index_map,subj_marker_indexes,obj_marker_indexes,const2id):
    f1,wp2const = get_wp2const_spans(wp_spans,const_spans,index_map,subj_marker_indexes,obj_marker_indexes)
    f2, const_seq = interpret_const_seq(seq,const2id,index_map,wp_ids,wp_spans,subj_marker_indexes,obj_marker_indexes)
    if wp2const[-1][-1] > 512 or len(const_seq) > 512:
        wp2const, const_seq = filter_long_results(wp2const,const_seq)
    return f1 and f2, wp2const,const_seq

def process(wp_fn,word_fn,const_fn,const_seq_fn,desc):
    N = len(const_fn)
    wp_to_const_spans = []
    const_seqs = []
    error_indexes = []
    for ix in trange(N,desc=desc):
        const_spans = const_fn[ix]
        seq = const_seq_fn[ix]
        wp_ids = wp_fn["wp_ids"][ix]
        wp_spans = wp_fn["spans"][ix]
        item = word_fn[ix]
        index_map = item["index_map"]
        if item["num_entities"] == 1:
            subj_marker_indexes = item["subj_obj_marker_indexes"]
            obj_marker_indexes = []
        else:
            subj_marker_indexes = item["subj_marker_indexes"]
            obj_marker_indexes = item["obj_marker_indexes"]
        f, tmp_wp2const, tmp_const_seq = per_example_process(const_spans,seq,wp_ids,wp_spans,index_map,subj_marker_indexes,obj_marker_indexes,const2id)
        if not f:
            error_indexes.append(ix)
        wp_to_const_spans.append(tmp_wp2const)
        const_seqs.append(tmp_const_seq)
    return wp_to_const_spans, {"seqs":const_seqs,"labels":wp_fn["labels"]}, error_indexes

if __name__ == "__main__":
    parser = ArgumentParser(description='Gererate ready-to-use data for CE- and CT- syntax-enhanced models.')
    parser.add_argument("--wp_file_dir",type=str,help="directory to wordpiece-level files.")
    parser.add_argument("--word_file_dir",type=str,help="directory to word-level files.")
    parser.add_argument("--const_parse_dir", default=None,type=str,help="path to constituency parsing files.")
    parser.add_argument("--output_dir",type=str,help="output directory.")
    args = parser.parse_args()

    #const2id = pickle.load(open(os.path.join(args.const_parse_dir,"const2id.pkl"),"rb"))
    # read and process train
    train_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"train.pkl"),"rb"))
    train_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"train.pkl"),"rb"))
    train_const_fn = pickle.load(open(os.path.join(args.const_parse_dir,"train_const_spans.pkl"),"rb"))
    train_const_seq_fn = pickle.load(open(os.path.join(args.const_parse_dir,"train_const_seqs.pkl"),"rb"))
    train_wp2const, train_const_seqs, train_error_indexes = process(train_wp_fn,train_word_fn,train_const_fn,train_const_seq_fn,"train")
    if len(train_error_indexes) != 0:
        print(f"{len(train_error_indexes)} errors on the train set.")
        pickle.dump(train_error_indexes,open(os.path.join(args.output_dir,"train_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    else:
        print(f"no errors on the train set.")
    pickle.dump(train_wp2const,open(os.path.join(args.output_dir,"train_wp2const.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_const_seqs,open(os.path.join(args.output_dir,"train_const_seqs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    # dev
    dev_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"dev.pkl"),"rb"))
    dev_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"dev.pkl"),"rb"))
    dev_const_fn = pickle.load(open(os.path.join(args.const_parse_dir,"dev_const_spans.pkl"),"rb"))
    dev_const_seq_fn = pickle.load(open(os.path.join(args.const_parse_dir,"dev_const_seqs.pkl"),"rb"))
    dev_wp2const, dev_const_seqs, dev_error_indexes = process(dev_wp_fn,dev_word_fn,dev_const_fn,dev_const_seq_fn,"dev")
    if len(dev_error_indexes) != 0:
        print(f"{len(dev_error_indexes)} errors on the dev set.")
        pickle.dump(dev_error_indexes,open(os.path.join(args.output_dir,"dev_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    else:
        print(f"no errors on the dev set.")
    pickle.dump(dev_wp2const,open(os.path.join(args.output_dir,"dev_wp2const.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump(dev_const_seqs,open(os.path.join(args.output_dir,"dev_const_seqs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

    # test
    test_wp_fn = pickle.load(open(os.path.join(args.wp_file_dir,"test.pkl"),"rb"))
    test_word_fn = pickle.load(open(os.path.join(args.word_file_dir,"test.pkl"),"rb"))
    test_const_fn = pickle.load(open(os.path.join(args.const_parse_dir,"test_const_spans.pkl"),"rb"))
    test_const_seq_fn = pickle.load(open(os.path.join(args.const_parse_dir,"test_const_seqs.pkl"),"rb"))
    test_wp2const, test_const_seqs, test_error_indexes = process(test_wp_fn,test_word_fn,test_const_fn,test_const_seq_fn,"test")
    if len(test_error_indexes) != 0:
        print(f"{len(test_error_indexes)} errors on the test set.")
        pickle.dump(test_error_indexes,open(os.path.join(args.output_dir,"test_error_indexes.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    else:
        print(f"no errors on the test set.")
    pickle.dump(test_wp2const,open(os.path.join(args.output_dir,"test_wp2const.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_const_seqs,open(os.path.join(args.output_dir,"test_const_seqs.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    
    print("finished.")