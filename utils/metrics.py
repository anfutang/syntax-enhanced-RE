import numpy as np
import torch
from scipy.stats import spearmanr
from collections import defaultdict

def correlation_distance(golds,preds,masks,correlations_per_length):
    # golds, preds: shape (N,Li,Li)
    for gold, pred, mask in zip(golds,preds,masks):
        length = mask.sum().item()
        target_ix = torch.where(mask.eq(0))[0]
        gold = torch.index_select(gold,0,target_ix)
        pred = torch.index_select(pred,0,target_ix)
        correlations_per_length[length].append(np.mean([spearmanr(g,p).correlation for g,p in zip(gold,pred)]))

def correlation_depth(golds,preds,masks,correlations_per_length):
    # golds, preds: shape (N,Li)
    for gold, pred, mask in zip(golds,preds,masks):
        length = mask.sum().item()
        target_ix = np.where(mask.eq(0))[0]
        gold = torch.index_select(gold,0,target_ix)
        pred = torch.index_select(pred,0,target_ix)
        correlations_per_length[length].append(spearmanr(gold,pred).correlation)
