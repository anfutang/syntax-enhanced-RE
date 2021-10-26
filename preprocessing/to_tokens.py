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

#===================== 
# OBJECTIVE: generate token-level data
# for each sentence, build a dictionary like: {"tokens" -> tokens containing markers, "dependency" -> (ix_1, ix2, dependency tag)}
# dependency are obtained using sentences without entity markers, but ix_1, ix_2 are words' indexes in original token sequences.
# overall pipeline: @@  @@ 