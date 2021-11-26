import os
import numpy as np
import pandas as pd
import logging
import torch
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

lb = MultiLabelBinarizer(classes=[0,3,4,5,6,9])
#m = torch.nn.Softmax(dim=1)

logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,path,name,mode,seed,batch_size,device):
        assert mode in ["probe_only","no_syntax","with_syntax"], "invalid mode: must be PROBE_ONLY, NO_SYNTAX, or WITH_SYNTAX"

        np.random.seed(seed)
        self.mode = mode
        self.device = device

        wp_data = pickle.load(open(os.path.join(path,f"wp_{name}.pkl"),"rb"))
        label_data = pickle.load(open(os.path.join(path,f"syntactic_probe_labels_{name}.pkl"),"rb"))
        wps = wp_data["wps"]

        if mode != "probe_only":
            labels = label_data["relations"]
            num_examples = len(labels)
        if mode != "no_syntax":
            maps = wp_data["map"]
            dist_matrixs = label_data["distance_matrix"]
            depths = label_data["depths"]
            masks = label_data["mask"]
            keys = label_data["keys"]
            num_examples = len(depths)

        if name == "train":
            indexes = list(range(num_examples))
            np.random.shuffle(indexes)
            wps = [wps[i] for i in indexes]
            if mode != "probe_only":
                labels = [labels[i] for i in indexes]
            if mode != "no_syntax":
                maps = [maps[i] for i in indexes]
                dist_matrixs = [dist_matrixs[i] for i in indexes]
                depths = [depths[i] for i in indexes]
                masks = [masks[i] for i in indexes]
                keys = [keys[i] for i in indexes]

        # order: wordpiece_ids, maps, keys, relation_labels, dist_matrix, depth_list
        if mode == "probe_only":
            self.data = list(zip(wps,maps,keys,masks,dist_matrixs,depths))
        else:
            if type(labels[0]) == str:
                #print(labels)
                labels = convert_labels(labels)
                #print('-'*10)
                #print(labels)
            if mode == "no_syntax":
                self.data = list(zip(wps,labels))
            else:
                self.data = list(zip(wps,maps,keys,masks,labels,dist_matrixs,depths))
        
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
        #print(list(unzip_batch[1]))
        #print(len(list(unzip_batch[1])))
        if self.mode == "no_syntax":
            assert len(unzip_batch) == 2, "mode NO_SYNTAX and input data do not match."
            return {"wps":wps,"labels":torch.Tensor(unzip_batch[1]).float().to(self.device)}
        elif self.mode == "probe_only":
            assert len(unzip_batch) == 6, "mode PROBE_ONLY and input data do not match."
            return {"wps":wps,
                    "maps":unzip_batch[1],
                    "keys":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[2]],
                    "masks":[torch.Tensor(lst).eq(0).to(self.device) for lst in unzip_batch[3]],
                    "dist_matrixs":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[4]],
                    "depths":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[5]]}
        else:
            assert len(unzip_batch) == 7, "mode WITH_SYNTAX and input data do not match."
            return {"wps":wps,
                    "maps":unzip_batch[1],
                    "keys":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[2]],
                    "masks":[torch.Tensor(lst).eq(0).to(self.device) for lst in unzip_batch[3]],
                    "labels":torch.Tensor(unzip_batch[4]).float().to(self.device),
                    "dist_matrixs":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[5]],
                    "depths":[torch.Tensor(lst).to(self.device) for lst in unzip_batch[6]]}

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def convert_labels(labels):
    return lb.fit_transform([list(map(int,l.split())) for l in labels]).tolist()
