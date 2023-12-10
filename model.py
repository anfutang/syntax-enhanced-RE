import logging
import numpy as np
import torch
from torch import nn
from torch.nn import (BCEWithLogitsLoss,CrossEntropyLoss)
from transformers import (BertPreTrainedModel, BertModel, BertLayer)
from utils import class_weights

logger = logging.getLogger(__name__)

class HighwayGateLayer(nn.Module):
    def __init__(self, dim, bias=True):
        super(HighwayGateLayer, self).__init__()
        self.transform = nn.Linear(dim, dim, bias=bias)

    def forward(self,v,z):
        # from sachan "do syntax trees help ...": equation (5)
        # v at the output of BERT, z at the output of syntax-GNN
        g = torch.sigmoid(self.transform(v))
        return g * v + (1 - g) * z

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        #print(hidden_states.shape)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SyntaxBert(BertPreTrainedModel):
    def __init__(self,config,args):
        assert args.model_type in ["no_syntax","extra","late_fusion","ce","ct"], "invalid model type; options allowed: no_syntax; extra; late_fusion; ce; ct; mts."

        super().__init__(config)
        self.num_labels = args.num_labels
        self.config = config
        self.model_type = args.model_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.clf = nn.Linear(config.hidden_size, config.num_labels)

        if args.model_type in ["late_fusion","ce","extra"]:
            self.pooler = BertPooler(config) 
            self.extra_layers = nn.ModuleList([BertLayer(config) for _ in range(args.num_extra_attention_layers)]) 
            self.highway = HighwayGateLayer(config.hidden_size)
            self.layernorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        
        self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[args.dataset_name])))     
        self.init_weights()

    def forward(self,input_ids=None,labels=None,token_type_ids=None,position_ids=None,output_attentions=None,output_hidden_states=None,
                attention_mask=None,wp2const=None,syntactic_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        if self.model_type in ["no_syntax","ct"]:
            pooled_output = outputs[1]
        else:
            hidden_states = outputs[0]
            if self.model_type == "ce":
                hidden_states, mask = self._get_chunk_embeddings(hidden_states,wp2const)
                prev_hidden_states = hidden_states
            elif self.model_type == "late_fusion":
                mask = syntactic_mask[:,None,:,:]
                prev_hidden_states = outputs[0]
                assert mask.shape[-1] == hidden_states.shape[1], "number of wordpieces not equal for hidden vectors and syntactic masks."
            else:
                prev_hidden_states = outputs[0]
                mask = attention_mask[:,None,None,:]
            mask = (1.0 - mask) * torch.finfo(torch.float).min
            for layer in self.extra_layers:
                hidden_states = layer(hidden_states,attention_mask=mask)[0]
            hidden_states = self.highway(prev_hidden_states,hidden_states)
            pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.clf(pooled_output)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1,self.num_labels),labels.view(-1,self.num_labels))
            return (loss, logits)
        else:
            return (logits,)

    def _get_chunk_embeddings(self,hidden_states,m):
        device = next(self.parameters()).device
        bs, _, dim = hidden_states.shape
        L = max(map(len,m))
        const_embs = torch.zeros(bs,L,dim)
        const_masks = torch.zeros(bs,L)
        for b, ma, emb in zip(list(range(bs)),m,hidden_states):
            # emb of shape (length, dim)
            ix = 0
            for start, end in ma:
                const_embs[b][ix] = emb[start:end].sum(dim=0)  
                ix += 1
            assert ix == len(ma), "number of chunks not right."
            const_masks[b][:len(ma)] = 1
        const_embs = const_embs.to(device)
        const_masks = const_masks.float().to(device) 
        const_embs = self.layernorm(const_embs)
        return const_embs, const_masks[:,None,None,:]

class MTSBert(BertPreTrainedModel):
    def __init__(self,config,args):
        assert args.model_type == "mts", "invalid model type; options allowed: no_syntax; extra; late_fusion; ce; ct; mts."

        super().__init__(config)
        self.num_labels = args.num_labels
        self.config = config
        self.alpha = args.alpha
        self.model_type = args.model_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.transformation_distance = nn.Linear(config.hidden_size,config.hidden_size)
        self.transformation_depth = nn.Linear(config.hidden_size,config.hidden_size)

        self.clf_re = nn.Linear(config.hidden_size, config.num_labels)
        self.clf_dist = nn.Linear(config.hidden_size,args.num_labels_of_probe_distance)
        self.clf_depth = nn.Linear(config.hidden_size,args.num_labels_of_probe_depth)

        self.layernorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        
        self.loss_re = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[args.dataset_name])))
        self.loss_dist = CrossEntropyLoss(weight=torch.Tensor(np.log(class_weights["probe_distance"])))  
        self.loss_depth = CrossEntropyLoss(weight=torch.Tensor(np.log(class_weights["probe_depth"])))     

        self.init_weights()
    
    def forward(self,input_ids=None,labels=None,dist_labels=None,depth_labels=None,token_type_ids=None,position_ids=None,
                output_attentions=None,output_hidden_states=None,attention_mask=None,wp2word=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.clf_re(pooled_output)
        if labels is None:
            return (logits,)
        loss_re = self.loss_re(logits.view(-1,self.num_labels),labels.view(-1,self.num_labels))
        hidden_states = self._get_word_embeddings(outputs[0],wp2word)
        # projection to linguistic space; respectively capturing distance and depth information
        dist_states = self.transformation_distance(hidden_states) # shape (B,L,d)
        depth_states = self.clf_depth(self.transformation_depth(hidden_states)) # shape (B,L,nc_depth)
        #print(dist_states.shape,depth_states.shape)
        dist_states = self.clf_dist((dist_states.unsqueeze(1)-dist_states.unsqueeze(2))**2) # shape (B,L,L,nc_dist)
        # flatten to calculate loss
        lengths = list(map(len,wp2word))
        dist_logits, depth_logits = self._flatten(dist_states,depth_states,lengths)
        loss_dist = self.loss_dist(dist_logits,dist_labels)
        loss_depth = self.loss_depth(depth_logits,depth_labels)
        loss =  loss_re + self.alpha * (loss_dist + loss_depth) / (1 + self.alpha)
        return (loss,logits)

    def _flatten(self,dist_states,depth_states,lengths):
        device = next(self.parameters()).device
        dist_logits, depth_logits = [], []
        for dist_state, depth_state, N in zip(dist_states,depth_states,lengths):
            depth_logits.append(depth_state[:N])
            for ix in range(N-1):
                dist_logits.append(dist_state[ix][ix+1:N])
        dist_logits = torch.vstack(dist_logits)
        depth_logits = torch.vstack(depth_logits)
        dist_logits = dist_logits.to(device)
        depth_logits = depth_logits.to(device)
        return dist_logits, depth_logits
    
    def _get_word_embeddings(self,hidden_states,m):
        device = next(self.parameters()).device
        bs, _, dim = hidden_states.shape
        L = max(map(len,m))
        word_embs = torch.zeros(bs,L,dim)
        for b, ma, emb in zip(list(range(bs)),m,hidden_states):
            # emb of shape (length, dim)
            ix = 0
            for start, end in ma:
                word_embs[b][ix] = emb[start:end].sum(dim=0)  
                ix += 1
            assert ix == len(ma), "number of words not right."
        word_embs = word_embs.to(device)
        word_embs = self.layernorm(word_embs)
        return word_embs
