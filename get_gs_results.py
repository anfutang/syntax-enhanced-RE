import os
import numpy as np
import pickle
from sklearn.metrics import f1_score
from argparse import ArgumentParser

# modify directly here the grid of hyperparameters that are used for the search

parameters = {"batch_size":[16,32],
              "learning_rate":[1e-4,1e-5,2e-5,5e-5]}

def vote(arr):
    voted_arr = []
    for a in arr:
        voted_arr.append((a==a.max()).astype("int"))
    return np.array(voted_arr)

def main(args):
    dataset = "test"
    if args.dev:
        dataset = "dev"
    labels = pickle.load(open(os.path.join(args.data_dir,f"syntactic_probe_labels_{dataset}.pkl"),"rb"))["relations"]

    scores = {}
    for bs in parameters["batch_size"]:
        for lr in parameters["learning_rate"]:
            preds_param = []
            for i in range(args.ensemble_size):
                model_path = os.path.join(args.model_dir,f"finetune_{args.mode}_{args.model_type}_{bs}_{lr}/seed_{args.base_seed+i}_ensemble_{i}")
                preds_tmp = np.array(pickle.load(open(os.path.join(model_path,"preds.pkl"),"rb")))
                preds_param.append(preds_tmp)
                scores[(bs,lr)] = f1_score(labels,preds_tmp,average="micro",labels=[1,2,3,4,5])
            preds_param = vote(sum(*preds_param))
            scores[(bs,lr)] = f1_score(labels,preds_param,average="micro",labels=[1,2,3,4,5])
    
    with open("./gs_scores.pkl","wb") as f:
        pickle.dump(scores,f,pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = ArgumentParser(description="this script collects grid search results and resume them in a dataframe.")
    parser.add_argument("--data_dir",type=str,required=True)
    parser.add_argument("--model_dir",type=str,required=True)
    parser.add_argument("--model_type",type=str,default="biobert")
    parser.add_argument("--mode",type=str,mode="no_syntax")
    parser.add_argument("--base_seed",type=int,default=42,help="seed from which to start") # we used base_seed+ensemble_id to set seeds for different models in an ensemble
    parser.add_argument("--ensemble_size",type=int,required=True,help="number of models in an ensemble")
    parser.add_argument("--dev",action="store_true",help="by default, evaluate on the test set; set this to evaluate on the validation set.")
    args = parser.parse_args()
    main(args)

