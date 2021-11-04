import torch

def syntactic_loss(output,dist_matrix,depths):
    # output: shape (N,Li,m), where Li corresponds to the number of tokens in tje i-th sentence, L is not a constant value among sentences.
    # dist_matrix: shape (N, Li, Li)
    # depths: shape (N, Li)
    return distance_loss(output,dist_matrix) + depth_loss(output,depths)

def distance_loss(output,target):
    dist_loss = 0
    for sent_embs, gold_matrix in zip(output,target):
        tmp_length = len(gold_matrix)
        pred_matrix = ((sent_embs.unsqueeze(1)-sent_embs.unsqueeze(0))**2).sum(-1)
        assert pred_matrix.shape == (tmp_length,tmp_length), "predicted distance matrix not in good shape."
        dist_loss += torch.mean((pred_matrix - gold_matrix**2))
    return dist_loss / len(target)

def depth_loss(output,target):
    depth_loss = 0
    for sent_embs, gold_depths in zip(output,target):
        tmp_length = len(gold_depths)
        pred_depths = (sent_embs**2).sum(-1)
        depth_loss += torch.mean((pred_depths - gold_depths**2))
    return depth_loss / len(target)

