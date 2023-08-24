import os
import pickle
from argparse import ArgumentParser

dist_map = {**{i+1:i for i in range(9)},
            **{i:9 for i in range(10,15)},
            **{i:10 for i in range(15,20)},
            **{i:11 for i in range(20,33)}}

depth_map = {**{i:i for i in range(6)},
              **{i:6 for i in range(6,10)},
              **{i:7 for i in range(10,25)}}

def remap(f):
    new_dists = []
    for dist in f["distances"]:
        new_dist = []
        for d in dist[:-1]:
            new_dist += [dist_map[dd] for dd in d[1:]]
        tmp_N = len(dist)
        assert len(new_dist) == tmp_N * (tmp_N-1) // 2, "invalid length."
        new_dists.append(new_dist)
    new_depths = []
    for depth in f["depths"]:
        new_depths.append([depth_map[d] for d in depth])
    return {"spans":f["spans"],"distances":new_dists,"depths":new_depths}  


def process(input_dir):
    train = pickle.load(open(os.path.join(input_dir,"train_probe.pkl"),"rb"))
    dev = pickle.load(open(os.path.join(input_dir,"dev_probe.pkl"),"rb"))
    test = pickle.load(open(os.path.join(input_dir,"test_probe.pkl"),"rb"))

    pickle.dump(remap(train),open(os.path.join(input_dir,"train_probe.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)
    pickle.dump(remap(dev),open(os.path.join(input_dir,"dev_probe.pkl","wb"),pickle.HIGHEST_PROTOCOL))
    pickle.dump(remap(test),open(os.path.join(input_dir,"test_probe.pkl"),"wb"),pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser(description='Gererate ready-to-use data for late-fusion syntax-enhanced models.')
    parser.add_argument("--dep_file_dir", default=None,type=str,help="path to dependency parsing files.")
    args = parser.parse_args()

    fn = os.path.join(args.dep_file_dir)


"""process("bbrel_full")
process("chemprot_blurb")
process("drugprot")"""