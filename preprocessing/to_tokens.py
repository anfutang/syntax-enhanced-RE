#== this script takes .csv files as input, the .csv files should be in format:
#== columns: "articleId", "arg1", "arg2", "sentence1", "sentence2" (optional), "label"
#== sentences should be contain entities marked like: "@@ SUBJECT @@", "$$ OBJECT $$", "¢¢ SUBJECT-OBJECT ¢¢" in cases where spans of the subject and the object overlap.
import os
import re
import argparse
import numpy as np
import pandas as pd
import stanza

parser = stanza.Pipeline("en",package="genia",verbose=False)

#===================== OVERVIEW ================== 
# OBJECTIVE: generate token-level data
# for each sentence, build a dictionary like: {"tokens" -> tokens containing markers, "dependency" -> (ix_1, ix2, dependency tag)}
# dependency are obtained using sentences without entity markers, but ix_1, ix_2 are words' indexes in original token sequences.
# overall pipeline: @@  @@ 
#===================== RULE-BASED TOKEN SEGMENTATION PROCESSING ======
# STEPS
# 1) obtain token sequences from the output of stanza
# 2) process compounds where targeted arguments are found within 
# NOTES
# A. why use stanza tokenization? 
# : I think parser tokenization is more accurate, if group only wordpieces to form tokens, sometimes the token segmentation is not much informative. 
#   For example, for the compound "[(6-iodo-2-methyl-1-[2-(4-morpholinyl)ethyl]-1H-indol-3-yl](4-methoxyphenyl)"
#   if only by grouping wordpieces, we get 36 tokens, but it is actually considered as one token by stanza.
#   for syntactic-enhanced networks, token segmentation from stanza is more informative because
#   (i) easier to describe dependencies if we use depedency-based information later in the network, e.g. adjacency matrix; 
#   (ii) tokenization serves as a kind of syntactic information as well.
# B. in which case we choose to seperate a compound?
# : in most cases (especially in the case of biomedical texts), compounds are chained nouns, e.g. 15-deoxy-Delta(12,14)-prostaglandin
#   for these compounds, there is no need to seperate them. Because our data is prepared for BERT, when applying bert-tokenizer on these tokens,
#   they will be tokenized into small pieces, in essence it is equivalent to manual split. The interest of seperating compounds are:
#   (i) to accentuate targeted arguments in cases where a targeted argument is within a compound
#   (ii) dependencies between subwords of compound may bring important information, but these noun-noun compounds are not this type. 
#        However, there do exist compounds that may be useful, for example, kainate-receptor-mediated, serum-starvation-induced, etc.


