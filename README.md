This repository hosts the code for the paper [Fine-grained Attention in Hierarchical Transformers for Tabular Time-series](https://arxiv.org/abs/2406.15327) by R. Azorin, Z. Ben Houidi, M. Gallo, A. Finamore, and P. Michiardi, accepted at [KDD'24 10th MiLeTS Workshop](https://kdd-milets.github.io/milets2024/).

_Fieldy_ is a fine-grained hierarchical Transformer that contextualizes fields at both the row and column levels. We compare our proposal against state of the art models on regression and classification tasks using public tabular time-series datasets. Our results show that combining row-wise and column-wise attention improves performance without increasing model size.

<img src="https://github.com/raphaaal/fieldy/blob/main/intro_fig.png" alt="intro_fig" width="500"/>

## Requirements
1. Create an environment with `conda create --name fieldy python==3.8.16` 
2. Activate it with `conda activate fieldy`
3. Install the requirements with `pip install -r requirements.txt`

## Datasets loading

Choose an option below to load the preprocessed datasets:

- With **DropBox**:
1. Use this [download link](https://www.dropbox.com/scl/fo/8wy5ng9t8nl60fwxkjpn5/ACK6X8d_O1XHGQwoPS4_OzA?rlkey=dym6nzfgzb1h7rqmdphogp4q5&st=0q7og92n&dl=0)
2. Move the downloaded files to:
      - `./data/kdd/KDDDataset_ft.pkl`
      - `./data/kdd/KDDDataset_ft.pkl`
      - `./data/prsa/PRSADataset_labeled.pkl`

- With the **raw CSVs**:
1. Delete the Git Large File Storage pointers:
      - `rm ./data/kdd/KDDDataset_ft.pkl`
      - `rm ./data/kdd/KDDDataset_ft.pkl`
      - `rm ./data/prsa/PRSADataset_labeled.pkl`
2. To preprocess the KDD dataset, use `./dataset/kdd.ipynb`.
3. To preprocess the PRSA dataset, you have nothing else to do, it will be automatically triggered.

## Models training

- To train and evaluate models on the **KDD Loan default prediction** dataset:
1. Run `chmod +x kdd.sh`
2. Run `./kdd.sh`

- To train and evaluate models on the **PRSA Beijing pollution** dataset:
1. Run `chmod +x prsa.sh`
2. Run `./prsa.sh`

Results will be saved under `./results`.

## Plot results
Use `./plots/results2latex.ipynb`.

## Toy task for field-wise attention
Use `./plots/field_wise_attention.ipynb`.

## Citation
If you use this paper or code as a reference, please cite it with:
```
@misc{azorin2024finegrained,
      title={Fine-grained Attention in Hierarchical Transformers for Tabular Time-series}, 
      author={Raphael Azorin and Zied Ben Houidi and Massimo Gallo and Alessandro Finamore and Pietro Michiardi},
      year={2024},
      eprint={2406.15327},
      archivePrefix={arXiv},
}
```

## Acknowledgements
This repository is built on top of [TabBERT](https://github.com/IBM/TabFormer).
We would also like to thanks the authors of [UniTTab](https://arxiv.org/abs/2302.06375), for discussions on metrics and pre-processing. 
