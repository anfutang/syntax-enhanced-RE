import numpy as np
import torch
from scipy.stats import spearmanr
from collections import defaultdict

def correlation_distance(golds,preds,masks,correlations_per_length):
    # golds, preds: shape (N,Li,Li)
    for gold, pred, mask in zip(golds,preds,masks):
        length = mask.sum()
        target_ix = np.where(mask)[0]
        gold = gold[target_ix,:][:,target_ix]
        pred = pred[target_ix,:][:,target_ix]
        correlations_per_length[length].append(np.mean([spearmanr(g,p).correlation for g,p in zip(gold,pred)]))

def correlation_depth(golds,preds,masks,correlations_per_length):
    # golds, preds: shape (N,Li)
    for gold, pred, mask in zip(golds,preds,masks):
        length = mask.sum()
        target_ix = np.where(mask)[0]
        gold = gold[target_ix]
        pred = pred[target_ix]
        correlations_per_length[length].append(spearmanr(gold,pred).correlation)
