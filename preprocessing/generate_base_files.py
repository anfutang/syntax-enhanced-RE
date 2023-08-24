import os
import pickle
from argparse import ArgumentParser

def process(args):
    train = pickle.load(open(os.path.join(args.input_dir,"train.pkl"),"rb"))
    dev = pickle.load(open(os.path.join(args.input_dir,"dev.pkl"),"rb"))
    test = pickle.load(open(os.path.join(args.input_dir,"test.pkl"),"rb"))

    pickle.dump({"wp_ids":train["wp_ids"],"labels":train["labels"]},open(os.path.join(args.output_dir,"train.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump({"wp_ids":dev["wp_ids"],"labels":dev["labels"]},open(os.path.join(args.output_dir,"dev.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump({"wp_ids":test["wp_ids"],"labels":test["labels"]},open(os.path.join(args.output_dir,"test.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = ArgumentParser(description='Data Preparation for Syntax-enhanced RE models: remove entity markers from sentences.')
    parser.add_argument("--input_dir", default=None, type=str,help="path to the directory containing word-level files.")
    parser.add_argument("--output_dir",default=None,type=str,help="directory to output. NOTE: should be an accessible directory.")
    args = parser.parse_args()

    process(args)
