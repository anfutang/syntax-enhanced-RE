import os
import numpy as np
import pandas as pd
import pickle
import stanza
import time
from tqdm import tqdm
from argparse import ArgumentParser

nlp = stanza.Pipeline(lang='en', processors='pos,lemma,tokenize,depparse', tokenize_pretokenized=True,package="CRAFT",logging_level='FATAL')

#====== this script perform dependency parsing on input sentences and save parsing results ======
# (this is the step after removing entity markers)
# input: sentences without entity markers 
# output: a list of lists containing triples (word_index_1, word_index_2, dependency relation)

def interpret_stanza_output(doc):
    # turn stanza outputs to list of triples (word_index_1, word_index_2, dependency relation)
    triples = [] # (word_index_1, word_index_2, dependency_relation)
    for word in doc.sentences[0].words:
        triples.append((word.id-1,word.head-1,word.deprel))
    return triples

if __name__ == "__main__":
    parser = ArgumentParser(description="Dependency parsing with biomedical Stanza (CRAFT).")
    parser.add_argument("--data_fn", default=None, type=str,help="path to word-level files. NOTE: should be an accessible filename (.pkl file)." 
                        "each value in the dictionary should contain keys: words, cleaned_words, index_map, num_entities, " 
                        "subj_marker_indexes, obj_marker_indexes, subj_head, obj_head.")
    parser.add_argument("--output_dir",default=None,type=str,help="directory to output. NOTE: should be an accessible directory.")
    parser.add_argument("--output_fn",type=str,help="output filename.")
    args = parser.parse_args()

    start_time = time.time()
    data = pickle.load(open(args.data_fn,"rb"))
    
    parse_res = {}
    for ix in tqdm(range(len(data))):
        cleaned_words = data[ix]["cleaned_words"]
        doc = nlp([cleaned_words])
        parse_res[ix] = interpret_stanza_output(doc)

    with open(os.path.join(args.output_dir,args.output_fn),"wb") as f:
        pickle.dump(parse_res,f,pickle.HIGHEST_PROTOCOL)
    end_time = time.time()
    print(f"time: {end_time-start_time} s.")
    print("finished.")


