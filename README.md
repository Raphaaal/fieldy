This repository hosts the code for the paper [Fine-grained Attention in Hierarchical Transformers for Tabular Time-series](https://arxiv.org/abs/2406.15327) by R. Azorin, Z. Ben Houidi, M. Gallo, A. Finamore, and P. Michiardi, accepted at [KDD'24 10th MiLeTS Workshop](https://kdd-milets.github.io/milets2024/).

_Fieldy_ is a fine-grained hierarchical Transformer that contextualizes fields at both the row and column levels. We compare our proposal against state of the art models on regression and classification tasks using public tabular time-series datasets. Our results show that combining row-wise and column-wise attention improves performance without increasing model size.

<img src="https://github.com/raphaaal/fieldy/blob/main/intro_fig.png" alt="intro_fig" width="500"/>

## Requirements
Run `conda create --name <env> --file requirements.txt`.

## Models training
Activate the conda environment and run `./kdd.sh` and `./prsa.sh`. 
Note that the pre-processed datasets are located at `./data/kdd/*.pkl` and `./data/prsa/*.pkl`. If you have trouble reading them, you can process data manually with `./dataset/kdd.ipynb`.

## Results
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

## Aknowledgements
This repository is built on top of [TabBERT](https://github.com/IBM/TabFormer).
We would also like to thanks the authors of [UniTTab](https://arxiv.org/abs/2302.06375), for discussions on metrics and pre-processing. 
