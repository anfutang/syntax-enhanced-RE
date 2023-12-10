from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Finetuning syntax-PubMedBERT models')
     
	# we always use PubMedBERT
    group = parser.add_argument_group('--bert_options')
    group.add_argument("--dataset_name", default=None, type=str,
                        help="The name of datasets. We use a fixed rule to search for relevant data depending on the syntax-enhanced model type.")
    group.add_argument("--model_type",required=True,type=str,help="must be one of the following:"
                                                            "no_syntax; extra; late_fusion; ce; ct; mts.")
    group.add_argument("--config_path", default="pubmedbert_config.json", type=str,help="Path to pre-trained config")
    group.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    group.add_argument("--num_labels",type=int,required=True,
                        help="number of layers in the last linear layer")
    group.add_argument("--num_labels_of_probe_distance",type=int,default=12,help="number of classes of pairwise word distances in the dependency tree.")
    group.add_argument("--num_labels_of_probe_depth",type=int,default=8,help="number of classes of word depths in the dependency tree.")
    group.add_argument("--dry_run",action="store_true",help="make a quick test with a small subset of data")
    group.add_argument("--number_of_examples_for_dry_run",type=int,default=50)
    group.add_argument("--run_id",type=int,default=0)
    group.add_argument("--num_ensemble",type=int,
                        help="number of repetitive experiments to get an ensemble result")
    group.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    group.add_argument("--learning_rate",type=float,default=2e-5)
    group.add_argument("--num_extra_attention_layers",type=int,default=0)
    group.add_argument("--alpha",type=float,help="alpha coefficient to control the importance of syntactic losses for MTS-PubMedBERT.")
    group.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="max gradient norm.")
    group.add_argument("--num_epoches",default=20,type=int,help="maximum number of epochs")
    group.add_argument("--no_test",action="store_true",help="set this during the hyperparameter search.")
    group.add_argument("--seed",type=int)
    group.add_argument("--shuffle_train",action="store_true",help="if set, shuffle the train set before training.")
    group.add_argument("--warmup",action="store_true",help="if set, use linear warmup scheduler for learning rate.")
    group.add_argument("--warmup_ratio",type=float,default=0.1)
    group.add_argument('--logging_steps', type=int, default=50,help="Log every X updates steps.")
    args = parser.parse_args()
    return args
