import os
import numpy as np
import pandas as pd
import logging
import torch
import pickle

logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,path,name,mode,seed,batch_size,device):
        assert mode in ["probe_only","no_syntax","with_syntax"], "invalid mode: must be PROBE_ONLY, NO_SYNTAX, or WITH_SYNTAX"

        np.random.seed(seed)
        self.mode = mode
        self.device = device

        if mode != "probe_only":
            self.labels = list(pd.read_csv(os.path.join(path,f"{name}.csv")).label.values)
            num_examples = len(self.labels)
        if mode != "no_syntax":
            wp_data = pickle.load(open(os.path.join(path,f"wp_{name}.pkl"),"rb"))
            syntactic_labels = pickle.load(open(os.path.join(path,f"syntactic_probe_labels_{name}.pkl"),"rb"))
            self.wps = wp_data["wps"]
            self.maps = wp_data["map"]
            self.dist_matrixs = syntactic_labels["distance_matrix"]
            self.depths = syntactic_labels["depths"]
            self.keys = syntactic_labels["keys"]
            num_examples = len(self.depths)

        if name == "train":
            indexes = list(range(num_examples))
            np.random.shuffle(indexes)
            if mode != "probe_only":
                self.labels = [self.labels[i] for i in indexes]
            if mode != "no_syntax":
                self.wps = [self.wps[i] for i in indexes]
                self.maps = [self.maps[i] for i in indexes]
                self.dist_matrixs = [self.dist_matrixs[i] for i in indexes]
                self.depths = [self.depths[i] for i in indexes]
                self.keys = [self.keys[i] for i in indexes]

        # order: wordpiece_ids, maps, keys, relation_labels, dist_matrix, depth_list
        if mode == "probe_only":
            self.data = list(zip(self.wps,self.maps,self.keys,self.dist_matrixs,self.depths))
        elif mode == "no_syntax":
            self.data = list(zip(self.wps,self.relation_labels))
        else:
            self.data = list(zip(self.wps,self.maps,self.keys,self.relation_labels,self.dist_matrixs,self.depths))
        
        self.data = [self.data[i:i+batch_size] for i in range(0,num_examples,batch_size)] 
        logger.info(f"{name}: {len(self.data)} batches generated.")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        unzip_batch = list(zip(*batch))
        
        wps = unzip_batch[0]
        max_length = max(map(len,wps))
        wps = torch.Tensor([wp + [0] * (max_length-len(wp)) for wp in wps]).long().to(self.device)
        #print(wps.shape)
        if self.mode == "no_syntax":
            assert len(unzip_batch) == 2, "mode NO_SYNTAX and input data do not match."
            return {"wps":wps,"labels":torch.Tensor([int(i) for i in unzip_batch[1]]).to(self.device)}
        elif self.mode == "probe_only":
            assert len(unzip_batch) == 5, "mode PROBE_ONLY and input data do not match."
            return {"wps":wps,
                    "maps":unzip_batch[1],
                    "keys":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[2]],
                    "dist_matrixs":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[3]],
                    "depths":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[4]]}
        else:
            assert len(unzip_batch) == 6, "mode WITH_SYNTAX and input data do not match."
            return {"wps":wps,
                    "maps":unzip_batch[1],
                    "keys":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[2]],
                    "labels":torch.Tensor([int(i) for i in unzip_batch[3]]).to(self.device),
                    "dist_matrixs":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[4]],
                    "depths":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[5]]}

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


