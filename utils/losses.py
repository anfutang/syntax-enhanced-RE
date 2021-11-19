import torch

def syntactic_loss(output,masks,dist_matrix,depths):
    # output: shape (N,Li,m), where Li corresponds to the number of tokens in tje i-th sentence, L is not a constant value among sentences.
    # dist_matrix: shape (N, Li, Li)
    # depths: shape (N, Li)
    return distance_loss(output,dist_matrix,masks) + depth_loss(output,depths,masks)

def distance_loss(output,target,masks):
    dist_loss = torch.zeros(1,dtype=torch.float,requires_grad=True).to(target[0].device)
    for sent_embs, gold_matrix, mask in zip(output,target,masks):
        #print(sent_embs.shape,gold_matrix.shape,mask.shape)
        #print(mask)
        assert sent_embs.shape[0] == gold_matrix.shape[0] == mask.shape[0], "sentence hidden state shape | gold distance matrix shape | punctuation mask shape NOT MATCHED."
        tmp_length = len(gold_matrix)
        pred_matrix = ((sent_embs.unsqueeze(1)-sent_embs.unsqueeze(0))**2).sum(-1)
        #print(pred_matrix.shape,gold_matrix.shape,sent_embs.shape)
        assert pred_matrix.shape[0] == tmp_length, "predicted distance matrix not in good shape."
        real_length = mask.eq(0).sum().item() # length of token sequences with punctuations removed
        #calculate the loss after removing punctuations
        dist_loss += torch.abs(pred_matrix - gold_matrix**2).masked_fill_(mask.unsqueeze(0),0).masked_fill_(mask.unsqueeze(1),0).sum() / (real_length ** 2) 
    #print(dist_loss)
    return dist_loss / len(target)

def depth_loss(output,target,masks):
    depth_loss = torch.zeros(1,dtype=torch.float,requires_grad=True).to(target[0].device)
    for sent_embs, gold_depths, mask in zip(output,target,masks):
        assert sent_embs.shape[0] == gold_depths.shape[0] == mask.shape[0], "sentence hidden state shape | gold depth shape | punctuation mask shape NOT MATCHED."
        tmp_length = len(gold_depths)
        pred_depths = (sent_embs**2).sum(-1)
        assert pred_depths.shape[0] == tmp_length, "predicted depths not in good shape."
        real_length = mask.eq(0).sum().item()
        depth_loss += torch.abs(pred_depths - gold_depths**2).masked_fill_(mask,0).sum() / real_length
    return depth_loss / len(target)

