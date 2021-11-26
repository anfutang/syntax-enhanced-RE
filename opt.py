from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Finetuning BERT models')
     
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                         help="The training data path. Should contain the .csv files")
    parser.add_argument("--force_cpu",action="store_true",
                         help="if set, the script will be run WITHOUT GPU.")
    parser.add_argument("--model_type",default="biobert",type=str,
                         help="abbreviation of model to use")
    parser.add_argument("--config_name_or_path", default="", type=str, 
                         help="Path to pre-trained config or shortcut name selected in the list")
    parser.add_argument("--dataset_name",type=str,default="chemprot",
                         help="name of the dataset that will be tested")
    parser.add_argument("--model_dir", default=None, type=str, 
                         help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir",default=None,type=str,
                         help="Path to predictions (set it only for inference)")
    parser.add_argument("--num_labels",type=int,default=2,
                         help="number of relations (no_relation COUNTED)")
    parser.add_argument("--debug",action="store_true",
                         help="use only first 100 examples to test the whole script.")
    parser.add_argument("--num_debug",type=int,default=100,
                         help="first num_debug examples will be used for a fast test.")
    parser.add_argument("--seed",type=int,default=42,
                         help="random seed for ensure reproducibility")
    parser.add_argument("--no_randomness",action="store_true",
                         help="if set, outputs of network are fixed.")
    parser.add_argument("--mode",type=str,default="probe_only",help="set the mode for finetuning model: {'probe_only','no_syntax','with_syntax'}.\n"
                                                                         "-probe_only: use only syntactic labels\n"
                                                                         "-no_syntax: finetune on RE task without introduing the syntax\n"
                                                                         "-with_syntax: syntax-driven finetune on RE task using extra syntactic loss items.")
    parser.add_argument("--probe_type",type=str,help="choose the type of probe; only set this option if mode is set to PROBE_ONLY. Under the mode WITH_SYNTAX,"
                                                     "all probe losses will be used.")
    parser.add_argument("--probe_only_no_train",action="store_true",help="only set this when probe_type is PROBE_ONLY; if set, raw representations from BERT will be used without training a linear classifier.")
    parser.add_argument("--save_predictions",action="store_true",help="set to save raw predictions on the chosen dataset.")
    parser.add_argument("--layer_index",type=int,default=12,help="indicate outputs of which layer in BERT to use.")
    parser.add_argument("--probe_rank",type=int,default=-1,help="the linear transformation rank for chosen syntactic probe.")
    parser.add_argument("--freeze_bert",action="store_true",help="if set, only task-specific layers for downstream tasks after BERT will be trained.:")    
    parser.add_argument("--grid_search",action="store_true",help="if set, results will be stored using a different format.")    

    parser.add_argument("--batch_size", default=32, type=int)

    group = parser.add_argument_group('--training_options')
    group.add_argument('--logging_steps', type=int, default=50,
                         help="Log every X updates steps.")
    group.add_argument("--monitor",type=str,default="score",
                         help="criteria to use for early stopping")
    group.add_argument("--early_stopping",action="store_true",
                         help="if use early stopping during training")
    group.add_argument("--max_num_epochs",default=10,type=int,
                         help="maximum number of epochs")
    group.add_argument("--num_train_epochs", default=3, type=int,
                         help="Total number of training epochs to perform.")
    group.add_argument("--patience",type=int,default=3,
                         help="patience of early stopping; if set, early stopping MUST be enabled.")
    group.add_argument("--max_seq_length", default=512, type=int,
                         help="The maximum input sequence length after tokenization. Sequences longer "
                              "than this will be truncated, sequences shorter will be padded.")
    group.add_argument("--ensemble_id",type=int,default=1,
                         help="number of repetitive experiments to get an ensemble result")
    group.add_argument("--learning_rate",type=float,default=2e-5)
    group.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm for gradient clipping.")
     
    args = parser.parse_args()
    return args
