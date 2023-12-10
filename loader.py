import os
import numpy as np
import pickle
import logging
import torch
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,args,tag,inference=False):
        # tag MUST BE in {"train","dev","test"}
        self.inference = inference
        self.tag = tag
        self.device = args.device
        self.model_type = args.model_type
        self.mlb = MultiLabelBinarizer(classes=list(range(args.num_labels)))

        data = self._load_data(args)
        
        logger.info(f"{len(data)} data length.")
        if args.dry_run:
            data = data[:args.number_of_examples_for_dry_run]
        
        # shuffle the data for training set        
        if args.shuffle_train and tag == "train":
            #np.random.seed(args.seed)
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            data = [data[i] for i in indices]
        
        data = [data[i:i+args.batch_size] for i in range(0,len(data),args.batch_size)]
        self.data = data
        logger.info(f"{tag}: {len(data)} batches generated.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]

        if self.model_type in ["no_syntax","extra","ct"]:
            if self.inference:
                batch_wp_ids = batch
            else:
                batch_wp_ids, batch_labels = list(zip(*batch))
        elif self.model_type == "ce":
            if self.inference:
                batch_wp_ids, batch_wp2const = list(zip(*batch))
            else:
                batch_wp_ids, batch_wp2const, batch_labels = list(zip(*batch))
        elif self.model_type == "late_fusion":
            if self.inference:
                batch_wp_ids, batch_adjs = list(zip(*batch))
            else:
                batch_wp_ids, batch_adjs, batch_labels = list(zip(*batch))
            batch_adjs = self._to_adj_matrix(batch_adjs)
        else:
            if self.inference:
                batch_wp_ids, batch_wp2word, batch_dists, batch_depths = list(zip(*batch))
            else:
                batch_wp_ids, batch_wp2word, batch_dists, batch_depths, batch_labels = list(zip(*batch))
            batch_dists, batch_depths = self._flatten_labels(batch_dists,batch_depths)
        
        batch_wp_ids, batch_masks = self._padding(batch_wp_ids)
        encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_masks}
        if self.model_type == "ce":
            encoding.update({"wp2const":batch_wp2const})
        elif self.model_type == "late_fusion":
            encoding.update({"syntactic_mask":batch_adjs})
        elif self.model_type == "mts":
            encoding.update({"wp2word":batch_wp2word,"dist_labels":batch_dists,"depth_labels":batch_depths})
        
        if not self.inference:
            batch_labels = self.mlb.fit_transform(batch_labels).astype(np.float32)
            encoding.update({"labels":torch.from_numpy(batch_labels).to(self.device)})
            
        return encoding

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _to_adj_matrix(self,adjs):
        bs = len(adjs)
        L = max(map(len,adjs))
        adj_matrix = torch.zeros(bs,L,L)
        for m, adj in zip(adj_matrix,adjs):
            for i, j in adj.items():
                m[i][j] = 1.0
        adj_matrix = adj_matrix.float().to(self.device)
        return adj_matrix

    def _flatten_labels(self,dists,depths):
        flattened_dists, flattened_depths = [], []
        for dist in dists:
            flattened_dists += dist
        for depth in depths:
            flattened_depths += depth
        flattened_dists = torch.Tensor(flattened_dists).long().to(self.device)
        flattened_depths = torch.Tensor(flattened_depths).long().to(self.device)
        return flattened_dists, flattened_depths

    def _padding(self,wp_ids):
        max_len = max(map(len,wp_ids))
        if max_len > 512:
            max_len = 512
            wp_ids = [line[:512] for line in wp_ids]	
        wp_ids = torch.Tensor([line[:max_len] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        padded_masks = torch.Tensor([len(line) * [1] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        return wp_ids, padded_masks
        
    def _load_data(self,args):
        if self.inference:
            if args.model_type in ["no_syntax","extra"]:
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                data = list(data["wp_ids"])
            elif args.model_type == "ct":
                data = pickle.load(open(f"./data/const_res/{args.dataset_name}/{self.tag}_const_seqs.pkl","rb"))
                data = list(data["seqs"])
            elif args.model_type == "ce":
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                wp2const = pickle.load(open(f"./data/const_res/{args.dataset_name}/{self.tag}_wp2const.pkl","rb"))
                data = list(zip(data["wp_ids"],wp2const))
            elif args.model_type == "late_fusion":
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                adjs = pickle.load(open(f"./data/dep_res/{args.dataset_name}/{self.tag}_adjs.pkl","rb"))
                data = list(zip(data["wp_ids"],adjs))
            else:
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                probe_data = pickle.load(open(f"./data/dep_res/{args.dataset_name}/{self.tag}_probe.pkl","rb"))
                data = list(zip(data["wp_ids"],probe_data["spans"],probe_data["distances"],probe_data["depths"]))
        else:
            if args.model_type in ["no_syntax","extra"]:
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                data = list(zip(data["wp_ids"],data["labels"]))
            elif args.model_type == "ct":
                data = pickle.load(open(f"./data/const_res/{args.dataset_name}/{self.tag}_const_seqs.pkl","rb"))
                data = list(zip(data["seqs"],data["labels"]))
            elif args.model_type == "ce":
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                wp2const = pickle.load(open(f"./data/const_res/{args.dataset_name}/{self.tag}_wp2const.pkl","rb"))
                data = list(zip(data["wp_ids"],wp2const,data["labels"]))
            elif args.model_type == "late_fusion":
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                adjs = pickle.load(open(f"./data/dep_res/{args.dataset_name}/{self.tag}_adjs.pkl","rb"))
                data = list(zip(data["wp_ids"],adjs,data["labels"]))
            else:
                data = pickle.load(open(f"./data/base_files/{args.dataset_name}/{self.tag}.pkl","rb"))
                probe_data = pickle.load(open(f"./data/dep_res/{args.dataset_name}/{self.tag}_probe.pkl","rb"))
                data = list(zip(data["wp_ids"],probe_data["spans"],probe_data["distances"],probe_data["depths"],data["labels"]))
        return data
