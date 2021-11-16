import numpy as np
import random
import logging
import torch
from torch import nn
from torch.nn import (BCELoss, BCEWithLogitsLoss)
from transformers import (BertPreTrainedModel, BertModel)
from utils.losses import syntactic_loss, distance_loss, depth_loss

logger = logging.getLogger(__name__)

def set_seed(args,ensemble_id=0):
    seed = args.seed + ensemble_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SyntaxBertModel(BertPreTrainedModel):
    def __init__(self,config,mode,layer_index=-1,probe_type=None,probe_rank=-1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mode = mode
        self.probe_type = probe_type
        self.layer_index = layer_index
        self.num_bert_layers = config.num_hidden_layers
        if probe_rank == -1:
            self.probe_dim = config.hidden_size
        else:
            self.probe_dim = probe_rank

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if mode == "no_syntax":
            self.pooler = BertPooler(config) 
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            #self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weights)) 
        elif mode == "probe_only":
            self.classifier = nn.Linear(config.hidden_size,config.probe_dim)          
 
        logger.info(f"SyntaxBERT loaded: mode={mode}; layer index={layer_index}; probe type={probe_type}; probe dimension={self.probe_dim}")

        self.init_weights()

    def forward(self,
                wps=None,
                maps=None,
                masks=None,
                keys=None,
                dist_matrixs=None,
                depths=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):
        if self.mode == "no_syntax":
            outputs = self.bert(wps,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                #loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1,self.num_labels))
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        elif self.mode == "probe_only":
            if self.layer_index == 0:
                sequence_output = self.bert.embeddings(input_ids=wps)
            else: # use the output of the indicated layer
                outputs = self.bert(input_ids=wps,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    output_hidden_states=True)
                assert len(outputs[2]) == self.num_bert_layers, "failed fetching hidden states from all BERT layers."
                sequence_output = outputs[2][self.layer_index-1]
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)
            #print(logits.shape)
            token_logits = []
            for logit, ma, key, mask in zip(logits,maps,keys,masks):
                token_logit = self._from_wps_to_token(logit,ma).to(next(self.parameters()).device)
                token_logits.append(torch.index_select(token_logit,0,key))
            if self.probe_type == "distance":
                loss = distance_loss(token_logits,dist_matrixs,mask)
            elif self.probe_type == "depth":
                loss = depth_loss(token_logits,depths,mask)
            output = (token_logits,)
            return ((loss,) + output) if loss is not None else output

    def _from_wps_to_token(self,wp_embs,span_indexes):
        curr_ix = 0
        token_embs = []
        for i in span_indexes:
            token_embs.append(torch.mean(wp_embs[curr_ix:i,:],dim=0).unsqueeze(0))
            curr_ix = i
        return torch.cat(token_embs,dim=0)
