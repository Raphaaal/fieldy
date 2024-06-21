This repository hosts the code for the paper [Fine-grained Attention in Hierarchical Transformers for Tabular Time-series](arxiv).

<img src="https://github.com/raphaaal/fieldy/blob/main/intro_fig.png" alt="intro_fig" width="500"/>

_Fieldy_ is a fine-grained hierarchical Transformer that contextualizes fields at both the row and column levels. We compare our proposal against state of the art models on regression and classification tasks using public tabular time-series datasets. Our results show that combining row-wise and column-wise attention improves performance without increasing model size.

# Requirements
Run `conda create --name <env> --file requirements.txt`.

# Models training
Activate the conda environment and run `./kdd.sh`  and `./prsa.sh`.

# Display results :
Use `./plots/results2latex.ipynb`.

# Field-wise attention toy task
Use `./plots/field_wise_attention.ipynb`.

# Cite
If you use this paper or code as a reference, please cite it with:
```
{

}
```

# Aknowledgements
This repository is built on top of [TabBERT](https://github.com/IBM/TabFormer).
We would also like to thanks the authors of [UniTTab](https://arxiv.org/abs/2302.06375), for discussions on metrics and pre-processing. 
