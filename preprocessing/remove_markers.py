import os
import numpy as np
import pandas as pd
import pickle
import stanza
import time
from tqdm import tqdm
from argparse import ArgumentParser

# we use the CRAFT biomedical model
nlp = stanza.Pipeline(lang='en', package="CRAFT", processors='pos,lemma,tokenize,depparse', tokenize_pretokenized=True,logging_level='FATAL')

def get_syntactic_head(noun,stanza_parser):
    doc = stanza_parser([noun])
    for word in doc.sentences[0].words:
        if word.deprel == "root":
            return word.id
        
def check(words,cleaned_words,m):
    for i, j in m.items():
        if cleaned_words[i] != words[j]:
            return False
    return True
        
def clean_with_two_entities(words,stanza_parser):
    cleaned_words = []
    subj_indexes, obj_indexes = [], []
    m = {}
    # m is a map from index in the sentence with markers removed -> index in the original sentence
    # e.g. give a list of words ['[CLS]','a','@','@','b','f','[SEP]']; m will be {1:1,2:4,3:5}.
    num_markers = 0
    for i, w in enumerate(words):
        if w in ["[CLS]","[SEP]"]:
            continue
        if w in ['@','$']:
            num_markers += 1
            if w == '@':
                subj_indexes.append(i)
            else:
                obj_indexes.append(i)
        else:
            m[i-num_markers-1] = i
            cleaned_words.append(w)
    # calculate the syntactic head of subject & object entities;
    subj_entity = words[subj_indexes[1]+1:subj_indexes[2]]
    obj_entity = words[obj_indexes[1]+1:obj_indexes[2]]
    if len(subj_entity) == 1:
        subj_head = subj_indexes[1] + 1
    else:
        subj_head = subj_indexes[1] + get_syntactic_head(subj_entity,stanza_parser)
    if len(obj_entity) == 1:
        obj_head = obj_indexes[1] + 1
    else:
        obj_head = obj_indexes[1] + get_syntactic_head(obj_entity,stanza_parser)
        
    # check if m is right; 
    t = check(words,cleaned_words,m) and len(cleaned_words) == len(words) - 10
    # check if subj and obj are correctly indexed;
    t = t and set([words[i] for i in subj_indexes]) == set(['@'])
    t = t and set([words[i] for i in obj_indexes]) == set(['$'])
    return t, cleaned_words, m, subj_indexes, obj_indexes, subj_head, obj_head

def clean_with_single_entity(words,stanza_parser):
    cleaned_words = []
    subj_obj_indexes = []
    m = {}
    num_markers = 0
    for i, w in enumerate(words):
        if w in ["[CLS]","[SEP]"]:
            continue
        if w == "¢¢":
            num_markers += 1
            subj_obj_indexes.append(i)
        else:
            m[i-num_markers-1] = i
            cleaned_words.append(w)
    # calculate the syntactic head of subject & object entities;
    subj_obj_entity = words[subj_obj_indexes[0]+1:subj_obj_indexes[1]]
    if len(subj_obj_entity) == 1:
        subj_obj_head = subj_obj_indexes[0] + 1
    else:
        subj_obj_head = subj_obj_indexes[0] + get_syntactic_head(subj_obj_entity,stanza_parser)
        
    # check if m is right; 
    t = check(words,cleaned_words,m) and len(cleaned_words) == len(words) - 4
    # check if subj and obj are correctly indexed;
    t = t and set([words[i] for i in subj_obj_indexes]) == set(['¢¢'])
    return t, cleaned_words, m, subj_obj_indexes, subj_obj_head

def clean_for_chemprot_with_two_entities(words):
    cleaned_words = []
    subj_indexes, obj_indexes = [], []
    m = {}
    num_markers = 0
    counter_markers = {'@':0,'$':0}
    for i, w in enumerate(words):
        if w in ["[CLS]","[SEP]"]:
            continue
        if w == '@':
            num_markers += 1
            if counter_markers[w] == 0:
                counter_markers[w] += 1
                subj_indexes.append(i)
            else:
                obj_indexes.append(i)
        elif w == '$':
            num_markers += 1
            if counter_markers[w] == 0:
                counter_markers[w] += 1
                subj_indexes.append(i)
            else:
                obj_indexes.append(i)
        else: 
            m[i-num_markers-1] = i
            cleaned_words.append(w)
    # for chemprot, subject & object entities are normalised and therefore both contains one word.
    subj_entity = words[subj_indexes[0]+1:subj_indexes[1]]
    obj_entity = words[obj_indexes[0]+1:obj_indexes[1]]
    assert len(subj_entity) == 1 and len(obj_entity) == 1, \
            "for chemprot subject & object entity both contain a single word."
    subj_head = subj_indexes[0] + 1
    obj_head = obj_indexes[0] + 1
    # check if m is right; 
    t = check(words,cleaned_words,m) and len(cleaned_words) == len(words) - 6
    # check if subj and obj are correctly indexed;
    t = t and (words[subj_indexes[0]] == '@' and words[subj_indexes[1]] == '$')
    t = t and (words[obj_indexes[0]] == '@' and words[obj_indexes[1]] == '$')
    # for chemprot, marked entities must be 'chemical' or 'gene'
    t = t and subj_entity[0] in ["chemical","gene"] and obj_entity[0] in ["chemical","gene"]
    return t, cleaned_words, m, subj_indexes, obj_indexes, subj_head, obj_head

def clean_for_chemprot_with_single_entity(words):
    cleaned_words = []
    subj_obj_indexes = []
    m = {}
    num_markers = 0
    for i, w in enumerate(words):
        if w in ["[CLS]","[SEP]"]:
            continue
        if w in ['@','$']:
            num_markers += 1
            subj_obj_indexes.append(i)
        else:
            m[i-num_markers-1] = i
            cleaned_words.append(w)
    subj_obj_entity = words[subj_obj_indexes[0]+1:subj_obj_indexes[1]]
    assert len(subj_obj_entity) == 3, \
        "for chemprot the subject-object entity must be: 'chemical', '-', 'gene'."
    subj_obj_head = subj_obj_indexes[0] + 1
    t = check(words,cleaned_words,m) and len(cleaned_words) == len(words) - 4
    t = t and words[subj_obj_indexes[0]] == '@' and words[subj_obj_indexes[1]] == '$'
    t = t and subj_obj_entity == ["chem",'-',"gene"]
    return t, cleaned_words, m, subj_obj_indexes, subj_obj_head

def clean_word_list(words,stanza_parser):
    # this function turn a list of words to a cleaned version with the entity markers, [CLS], [SEP] removed.
    # return:
    # 1) map from new index to old index;
    # 2) cleaned list of words;
    # 3) subject and object marker index;
    # 4) OLD index of the syntactic head of the subject & object (or the subject-object) entities (entity);
    # input must be ready for BERT: [CLS] at the first position and [SEP] at the last position.
    assert words[0] == "[CLS]" and words[-1] == "[SEP]", "[CLS] or [SEP] missing."
    # case#1: two entities; treat word lists like [...'@','@',W1,W2,'@','@',...,'$','$',W3,W4,W5,'$','$',...]
    if words.count('@') == 4 and words.count('$') == 4:
        return clean_with_two_entities(words,stanza_parser)
    # case#2: one entity; treat word lists like [...'¢¢',W1,W2,'¢¢']. Note that PubMedBERT tokenizes '¢¢' to '¢', '##¢'
    if words.count('¢¢') == 2:
        return clean_with_single_entity(words,stanza_parser)
    # case#3: specially for ChemProt (Blurb) with two entities; treat word lists like [...'@',W1,W2,'$',...,'@',W3,W4,W5,'$']
    if words.count('@') == 2 and words.count('$') == 2:
        return clean_for_chemprot_with_two_entities(words)
    # case#4: specially for ChemProt (Blurb) with one entity; treat word lists like [...'@',W1,W2,'$',...]
    if words.count('@') == 1 and words.count('$') == 1:
        return clean_for_chemprot_with_single_entity(words)
    return (False,)

# this function treats examples in the BB-Rel test set on which errors occured; they are only errors occured in the three datasets that we tested.
# It is not directly added into the clean() function for the purpose of possible usage of other dataset. 
# Error indexes are saved and you need to check why errors occured.  

"""def clean_bbrel_exceptions(words,stanza_parser):
    cleaned_words = []
    subj_indexes, obj_indexes = [], []
    m = {}
    num_markers = 0
    for i, w in enumerate(words):
        if w in ["[CLS]","[SEP]"]:
            continue
        if w == '@':
            num_markers += 1
            subj_indexes.append(i)
        elif w == '$' and ((i >= 1 and words[i-1] == '$') or (i < len(words) - 1 and words[i+1] == '$')):
            num_markers += 1
            obj_indexes.append(i)
        else:
            m[i-num_markers-1] = i
            cleaned_words.append(w)
            
    subj_entity = words[subj_indexes[1]+1:subj_indexes[2]]
    obj_entity = words[obj_indexes[1]+1:obj_indexes[2]]
    if len(subj_entity) == 1:
        subj_head = subj_indexes[1] + 1
    else:
        subj_head = subj_indexes[1] + get_syntactic_head(subj_entity,stanza_parser)
    if len(obj_entity) == 1:
        obj_head = obj_indexes[1] + 1
    else:
        obj_head = obj_indexes[1] + get_syntactic_head(obj_entity,stanza_parser)
        
    t = check(words,cleaned_words,m) and len(cleaned_words) == len(words) - 10
    t = t and set([words[i] for i in subj_indexes]) == set(['@'])
    t = t and set([words[i] for i in obj_indexes]) == set(['$'])
    return t, cleaned_words, m, subj_indexes, obj_indexes, subj_head, obj_head"""

def clean(words_lst,stanza_parser):
    error_ixs = []
    res = {}
    for ix in tqdm(range(len(words_lst))):
        words = words_lst[ix]
        tmp = clean_word_list(words,stanza_parser)
        if len(tmp) == 1:
            assert not tmp[0], "! error occurs but identified as normal."
            print("ERROR-0: abnormal number of entity markers.")
            error_ixs.append([ix,0])
            res[ix] = {}
        elif len(tmp) == 5:
            t, cleaned_words, m, subj_obj_marker_indexes, subj_obj_head = tmp
            if not t:
                print("ERROR-1: error occurs; single entity.")
                error_ixs.append([ix,1])
                res[ix] = {}
            else:
                res[ix] = {"words":words,"cleaned_words":cleaned_words,"index_map":m,
                            "num_entities":1,"subj_obj_marker_indexes":subj_obj_marker_indexes,
                            "subj_obj_head":subj_obj_head}
        elif len(tmp) == 7:
            t, cleaned_words, m, subj_marker_indexes, obj_marker_indexes, subj_head, obj_head = tmp
            if not t:
                print("ERROR-2: error occurs; two entities.")
                error_ixs.append([ix,1])
                res[ix] = {}
            else:
                res[ix] = {"words":words,"cleaned_words":cleaned_words,"index_map":m,
                           "num_entities":2,"subj_marker_indexes":subj_marker_indexes,
                           "obj_marker_indexes":obj_marker_indexes,"subj_head":subj_head,"obj_head":obj_head}
        else:
            print("! abnormal number of elements in outputs.")
    return res, error_ixs

if __name__ == "__main__":
    parser = ArgumentParser(description='Data Preparation for Syntax-enhanced RE models: remove entity markers from sentences.')
    parser.add_argument("--data_fn", default=None, type=str,help="path to wordpiece files. NOTE: should be an accessible filename (.pkl file)." 
                        "should contain keys: wps, wp_ids, labels, spans, words.")
    parser.add_argument("--output_dir",default=None,type=str,help="directory to output. NOTE: should be an accessible directory.")
    parser.add_argument("--output_fn",type=str,help="output filename.")
    parser.add_argument("--error_output_fn",type=str,help="output filename to store indexes of examples on which errors occured.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    doc_wp = pickle.load(open(args.data_fn,"rb"))
    # doc cleaned is a list of directories. each directory contains following keys:
    # 1) either "words", "clean_words", "index_map", "num_entities", "subj_marker_indexes", "obj_marker_indexes", "subj_head", "obj_head" (two entities);
    # 2) either "words", "clean_words", "index_map", "num_entities", "subj_obj_marker_indexes", "subj_obj_head" (single entity).
    start_time = time.time()
    doc_cleaned, error_indexes = clean(doc_wp["words"],nlp)
    if len(error_indexes) > 0:
        pickle.dump(error_indexes,open(os.path.join(args.output_dir,args.error_output_fn),"wb"),pickle.HIGHEST_PROTOCOL)
        print(f"{len(error_indexes)} errors.")
    else:
        print("No Error.")
    pickle.dump(doc_cleaned,open(os.path.join(args.output_dir,args.output_fn),"wb"),pickle.HIGHEST_PROTOCOL)
    end_time = time.time()
    print(f"time: {end_time-start_time} s.")
    print("finished.")
