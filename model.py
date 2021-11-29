import numpy as np
import random
import logging
import torch
from torch import nn
from torch.nn import (BCEWithLogitsLoss,CrossEntropyLoss)
from transformers import (BertPreTrainedModel, BertModel)
from utils.losses import syntactic_loss, distance_loss, depth_loss
from utils.constant import class_weights

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
    def __init__(self,config,mode,dataset_name,num_labels=2,layer_index=-1,probe_type=None,probe_rank=-1,train_probe=True,syntactic_coef=1):
        super().__init__(config)
        self.num_labels = num_labels
        self.mode = mode
        self.probe_type = probe_type
        self.layer_index = layer_index
        self.train_probe = train_probe
        self.syntactic_coef = syntactic_coef
        self.num_bert_layers = config.num_hidden_layers
        if probe_rank == -1:
            self.probe_dim = config.hidden_size
        else:
            self.probe_dim = probe_rank

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if mode == "no_syntax":
            self.pooler = BertPooler(config) 
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
            if dataset_name in ["chemprot"]:
                #print("yes, chemprot!")
                self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[dataset_name]))) # log(N/Nc)
                #self.loss_fct = BCEWithLogitsLoss()
            else:
                self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[dataset_name]))) 
        elif mode == "probe_only":
            self.classifier = nn.Linear(config.hidden_size,self.probe_dim)
        else:
            self.clf = nn.Linear(config.hidden_size,self.num_labels)
            self.clf_dist = nn.Linear(config.hidden_size,self.probe_dim)
            self.clf_depth = nn.Linear(config.hidden_size,self.probe_dim)
            self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[dataset_name])))          
 
        logger.info(f"SyntaxBERT loaded for dataset {dataset_name}: num of labels={num_labels}; mode={mode}; layer index={layer_index}; " + \
                    f"probe type={probe_type}; probe dimension={self.probe_dim}")

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
                labels=None,
                predict_only=False):
        if self.mode == "no_syntax":
            outputs = self.bert(wps,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            if not predict_only:
                assert labels is not None, "relation labels are NOT given."
                #print(logits.shape,labels.shape)
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1,self.num_labels))
                return (loss, logits)
            else:
                return (logits,) 
        elif self.mode == "probe_only":
            # use the output of the indicated layer
            outputs = self.bert(input_ids=wps,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    output_hidden_states=True)
            assert len(outputs[2]) == self.num_bert_layers + 1, "failed fetching hidden states from all BERT layers."
            sequence_output = outputs[2][self.layer_index]
            if self.train_probe:
                sequence_output = self.dropout(sequence_output)
                logits = self.classifier(sequence_output)
            else:
                logits = sequence_output
            token_logits = self._get_token_embs(logits,maps,keys)
            if not predict_only:
                if self.probe_type == "distance":
                    loss = distance_loss(token_logits,dist_matrixs,masks)
                elif self.probe_type == "depth":
                    loss = depth_loss(token_logits,depths,masks)
                return (loss, token_logits)
            else:
                return (token_logits,)
        else:
            outputs = self.bert(input_ids=wps,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids)
            sequence_output = self.dropout(outputs[0])
            pooled_output = self.dropout(outputs[1])
            logits = self.clf(pooled_output)

            #use three linear classifiers for three loss functions
            token_logits_dist = self._get_token_embs(self.clf_dist(pooled_output),maps,keys)
            token_logits_depth = self._get_token_embs(self.clf_depth(pooled_output),maps,keys)

            if not predict_only:
                # relation classification loss + distance probe loss + depth probe loss
                re_loss = self.loss_fct(logits.view(-1,self.num_labels),labels.view(-1,self.num_labels))
                syntactic_loss = distance_loss(token_logits_dist,dist_matrixs,masks) + depth_loss(token_logits_depth,depths,masks)
                loss = (re_loss + self.syntactic_coef * torch.sqrt(syntactic_loss)) / (1 + self.syntactic_coef)
                return (loss, logits)
            else:
                return (logits,) 
    
    def _get_token_embs(self,wp_embs,maps,keys):
        token_logits = []
        for logit, ma, key in zip(wp_embs,maps,keys):
            token_logit = self._from_wps_to_token(logit,ma).to(next(self.parameters()).device)
            token_logits.append(torch.index_select(token_logit,0,key))
        return token_logits

    def _from_wps_to_token(self,wp_embs,span_indexes):
        curr_ix = 0
        token_embs = []
        for i in span_indexes:
            token_embs.append(torch.mean(wp_embs[curr_ix:i,:],dim=0).unsqueeze(0))
            curr_ix = i
        return torch.cat(token_embs,dim=0)
