import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

def correlation_distance(golds,preds,correlations_per_length):
    # golds, preds: shape (N,Li,Li)
    for gold, pred in zip(golds,preds):
        #print(gold.shape,pred.shape)
        #print(gold)
        #print(pred)
        length = len(gold)
        correlations_per_length[length].append(np.mean([spearmanr(g,p).correlation for g,p in zip(gold,pred)]))

def correlation_depth(golds,preds,correlations_per_length):
    # golds, preds: shape (N,Li)
    for gold, pred in zip(golds,preds):
        length = len(gold)
        correlations_per_length[length].append(spearmanr(gold,pred).correlation)
