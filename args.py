import argparse


def define_main_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int,
        default=9,
        help="seed to use: 9[default]"
    )
    parser.add_argument(
        "--fieldtransf-nheads", 
        type=int,
        default=12,
        help="Nb of heads for TabBERT FieldTransformer"
    )
    parser.add_argument(
        "--fieldtransf-nlayers", 
        type=int,
        default=1,
        help="Nb of layers for TabBERT FieldTransformer"
    )
    parser.add_argument(
        "--dry-run", 
        type=int,
        default=0,
        help="Max number of samples in the full dataset"
    )
    parser.add_argument(
        "--trash", 
        action='store_true',
        help="Store results in /_trash/ folder (for debugging)"
    )
    parser.add_argument(
        "--scale-targets", 
        action='store_true',
        help="MinMax scale the targets for regression (e.g;, Pollution dataset)"
    )
    parser.add_argument(
        "--scaling", 
        type=str,
        default="std",
        choices=["std", "quantnorm", "minmax"],
        help="Type of target scaling performed"
    )
    parser.add_argument(
        "--runs", 
        type=int,
        default=1,
        help="Various runs, increasing the seed by 1"
    )
    parser.add_argument(
        "--n-layers", 
        type=int,
        default=12,
        help="Number of Sequence Transformer layers"
    )
    parser.add_argument(
        "--n-heads", 
        type=int,
        default=12,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--family", 
        choices=["fttransf_flatten", "tabbie", "column_tabbert", "row_tabbert", "fieldy"],
        help="Model family"
    )
    parser.add_argument(
        "--columnbert", 
        action='store_true',
        help="TabFormer but by columns, not rows, for the Fieldtransformer"
    )
    parser.add_argument(
        "--mlm", 
        action='store_true',
        help="masked lm loss; pass it for BERT",
        default=False,
    )
    parser.add_argument(
        "--mlm-loss", 
        type=str,
        help="Compute the loss across the attribute vocab or across all vocab [attribute, all]",
        default="all",
    )
    parser.add_argument(
        "--mlm-prob", 
        type=float,
        default=0.15,
        help="Token masking probability for MLM pre-training"
    )
    parser.add_argument(
        "--init-lr", 
        type=float,
        default=0.00005, # HF default 5e-05
        help="Initial pre-training lr"
    )
    parser.add_argument(
        "--ft-lr", 
        type=float,
        default=0.00005, # HF default 5e-05
        help="Initial fine-tuning lr"
    )
    parser.add_argument(
        "--dropout", 
        type=float,
        default=0.1, # HF default 0.1
        help="Dropout in BERT-like model"
    )
    parser.add_argument(
        "--data-type", 
        type=str,
        default="card", 
        choices=['card', 'prsa', 'kdd'],
        help='root directory for files'
    )
    parser.add_argument(
        "--data_root", 
        type=str,
        default="./data/credit_card/",
        help='root directory for files'
    )
    parser.add_argument(
        "--data_fname", 
        type=str,
        default="card_transaction.v1",
        help='file name of transaction'
    )
    parser.add_argument(
        "--data_extension", 
        type=str,
        default="",
        help="file name extension to add to cache"
    )
    parser.add_argument(
        "--vocab_file", 
        type=str,
        default='vocab.nb',
        help="cached vocab file"
    )
    parser.add_argument(
        '--user_ids', 
        nargs='+',
        default=None,
        help='pass list of user ids to filter data by'
    )
    parser.add_argument(
        "--cached", 
        action='store_true',
        help='use cached data files'
    )
    parser.add_argument(
        "--nrows", 
        type=int,
        default=None,
        help="no of transactions to use"
    )
    parser.add_argument(
        "--vocab-dir", 
        type=str,
        default='./data/',
        help="path to vocab load/dump"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default='checkpoints',
        help="path to model dump"
    )
    parser.add_argument(
        "--output-dir-initial", 
        type=str,
        default='checkpoints',
        help="path to model dump (used when they are multiple seed runs)"
    )
    parser.add_argument(
        "--pre-train", 
        action='store_true',
        help="Perform pretraining",
        # default=True,
    )
    parser.add_argument(
        "--fine-tune", 
        action='store_true',
        help="Perform fine-tuning",
        # default=True,
    )
    parser.add_argument(
        "--do-eval", 
        action='store_true',
        help="enable evaluation flag",
        default=True,
    )
    parser.add_argument(
        "--do-train", 
        action='store_true',
        help="enable training flag",
        default=True,
    )
    parser.add_argument(
        "--pos-emb",
        action='store_true',
        help="Embed the row position in the sequence",
    )
    parser.add_argument(
        "--col-emb", 
        action='store_true',
        help="Embed the column index for each field"
    )
    parser.add_argument(
        "--pt-epochs", 
        type=int,
        default=12,
        help="number of pretraining training epochs",
    )
    parser.add_argument(
        "--ft-epochs", 
        type=int,
        default=10,
        help="number of fine-tuning epochs"
    )
    parser.add_argument(
        "--hidden-size", 
        type=int,
        default=768,
        help="Hidden size for Sequence Transformer (must be divisible by the number of columns)"
    )
    parser.add_argument(
        "--bs", 
        type=int,
        default=120,
        help="Batch size",
    )
    parser.add_argument(
        "--mse", 
        action='store_true',
        help="Use MSE loss instead of BCE loss for multi-regression task",
        default=False,
    )

    return parser
