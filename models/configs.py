from transformers import BertConfig


class CustomBertConfig(BertConfig):
    def __init__(
        self,
        family=None,
        ncols=None,
        vocab_size=None,
        hidden_size=None,
        pad_token_id=None,
        seq_len=None,
        pos_emb=None,
        col_emb=None,
        fieldtransf_nheads=None,
        fieldtransf_nlayers=None,
        n_layers=None,
        insert_sep=None,
        num_ft_labels=None,
        dropout=None,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_token_id=pad_token_id, 
            **kwargs
        )
        self.family = family
        self.ncols = ncols
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pos_emb = pos_emb
        self.col_emb = col_emb
        self.fieldtransf_nheads = fieldtransf_nheads
        self.fieldtransf_nlayers = fieldtransf_nlayers
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.insert_sep = insert_sep
        self.num_ft_labels = num_ft_labels
        self.dropout = dropout


class FTTransformerFlattenConfig(CustomBertConfig):
    # Class attribute
    model_type = "FTTransformerFlatten"


class TabbieConfig(BertConfig):
    # Class attribute
    model_type = "Tabbie"
    def __init__(
        self,
        family=None,
        ncols=None,
        vocab_size=None,
        hidden_size=None,
        pad_token_id=None,
        seq_len=None,
        pos_emb=None,
        col_emb=None,
        fieldtransf_nheads=None,
        fieldtransf_nlayers=None,
        n_layers = None,
        insert_sep=None,
        num_ft_labels=None,
        dropout=None,
        **kwargs
    ):
        # Fine-tuning instantiaion
        if "tabbiefieldtransf_nlayers" in kwargs:
            tabbiefieldtransf_nlayers = kwargs["tabbiefieldtransf_nlayers"]
        # Pre-training instantiation
        else:
            tabbiefieldtransf_nlayers = fieldtransf_nlayers

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_token_id=pad_token_id, 
            **kwargs
        )
        self.family = family
        self.ncols = ncols
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pos_emb = pos_emb
        self.col_emb = col_emb
        self.fieldtransf_nheads = fieldtransf_nheads
        self.fieldtransf_nlayers = 1
        self.tabbiefieldtransf_nlayers = tabbiefieldtransf_nlayers
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.insert_sep = insert_sep
        self.num_ft_labels = num_ft_labels
        self.dropout = dropout


class ColumnTabBERTConfig(CustomBertConfig):
    # Class attribute
    model_type = "ColumnTabBERT"
  

class RowTabBERTConfig(CustomBertConfig):
    # Class attribute
    model_type = "RowTabBERT"
 

class FieldyConfig(CustomBertConfig):
    # Class attribute
    model_type = "Fieldy"
    
