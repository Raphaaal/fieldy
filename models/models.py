from transformers import BertTokenizer
from transformers.modeling_utils import PreTrainedModel

import torch
import torch.nn as nn

from models.modules import CustomBertForMaskedLM
from models.modules import ColumnTabBERTEmbeddings
from models.modules import RowTabBERTEmbeddings
from models.modules import TabFormerBertOnlyMLMHead

from models.configs import FTTransformerFlattenConfig
from models.configs import TabbieConfig
from models.configs import RowTabBERTConfig
from models.configs import ColumnTabBERTConfig
from models.configs import FieldyConfig


class Model:
    def __init__(
            self, 
            special_tokens, 
            vocab, 
            family=None,
            ncols=None, 
            hidden_size=None,
            seq_len=None,
            pos_emb=None,
            col_emb=None,
            max_position_embeddings=None,
            mlm_loss=None,
            n_layers=None,
            n_heads=None,
            fieldtransf_nheads=None,
            fieldtransf_nlayers=None,
            num_ft_labels=None,
            dropout=None,
        ):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename

        if family == "fttransf_flatten":
            self.config = FTTransformerFlattenConfig(
                family=family,
                vocab_size=len(self.vocab),
                ncols=self.ncols,
                hidden_size=hidden_size,
                seq_len=seq_len,
                pos_emb=pos_emb,
                col_emb=col_emb,
                pad_token_id=self.vocab.token2id["SPECIAL"]["[PAD]"][1],
                max_position_embeddings=max_position_embeddings,
                mlm_loss=mlm_loss,
                n_layers=n_layers,
                fieldtransf_nlayers=fieldtransf_nlayers,
                fieldtransf_nheads=fieldtransf_nheads,
                num_attention_heads=n_heads,
                num_ft_labels=num_ft_labels,
                dropout=dropout,
            )
        elif family == "tabbie":
            assert (hidden_size % ncols == 0)
            assert (hidden_size % seq_len == 0)
            self.config = TabbieConfig(
                family=family,
                vocab_size=len(self.vocab),
                ncols=self.ncols,
                hidden_size=hidden_size,
                seq_len=seq_len,
                pos_emb=pos_emb,
                col_emb=col_emb,
                pad_token_id=self.vocab.token2id["SPECIAL"]["[PAD]"][1],
                max_position_embeddings=max_position_embeddings,
                mlm_loss=mlm_loss,
                n_layers=n_layers,
                num_attention_heads=n_heads,
                fieldtransf_nlayers=fieldtransf_nlayers,
                fieldtransf_nheads=fieldtransf_nheads,
                num_ft_labels=num_ft_labels,
                dropout=dropout,
            )
        elif family == "column_tabbert":
            assert (hidden_size % seq_len == 0)
            assert (hidden_size % seq_len == 0)
            self.config = ColumnTabBERTConfig(
                family=family,
                vocab_size=len(self.vocab),
                ncols=self.ncols,
                hidden_size=hidden_size,
                seq_len=seq_len,
                pos_emb=pos_emb,
                col_emb=col_emb,
                pad_token_id=self.vocab.token2id["SPECIAL"]["[PAD]"][1],
                max_position_embeddings=max_position_embeddings,
                mlm_loss=mlm_loss,
                n_layers=n_layers,
                num_attention_heads=n_heads,
                fieldtransf_nlayers=fieldtransf_nlayers,
                fieldtransf_nheads=fieldtransf_nheads,
                num_ft_labels=num_ft_labels,
                dropout=dropout,
            )
        elif family == "row_tabbert":
            assert (hidden_size % ncols == 0)
            assert (hidden_size % ncols == 0)
            self.config = RowTabBERTConfig(
                family=family,
                vocab_size=len(self.vocab),
                ncols=self.ncols,
                hidden_size=hidden_size,
                seq_len=seq_len,
                pos_emb=pos_emb,
                col_emb=col_emb,
                pad_token_id=self.vocab.token2id["SPECIAL"]["[PAD]"][1],
                max_position_embeddings=max_position_embeddings,
                mlm_loss=mlm_loss,
                n_layers=n_layers,
                num_attention_heads=n_heads,
                fieldtransf_nlayers=fieldtransf_nlayers,
                fieldtransf_nheads=fieldtransf_nheads,
                num_ft_labels=num_ft_labels,
                dropout=dropout,
            )
        elif family == "fieldy":
            assert (hidden_size % ncols == 0)
            assert (hidden_size % seq_len == 0)
            self.config = FieldyConfig(
                family=family,
                vocab_size=len(self.vocab),
                ncols=self.ncols,
                hidden_size=hidden_size,
                seq_len=seq_len,
                pos_emb=pos_emb,
                col_emb=col_emb,
                pad_token_id=self.vocab.token2id["SPECIAL"]["[PAD]"][1],
                max_position_embeddings=max_position_embeddings,
                mlm_loss=mlm_loss,
                n_layers=n_layers,
                num_attention_heads=n_heads,
                fieldtransf_nlayers=fieldtransf_nlayers,
                fieldtransf_nheads=fieldtransf_nheads,
                num_ft_labels=num_ft_labels,
                dropout=dropout,
            )
        
        self.config.hidden_dropout_prob = self.config.dropout # Propagate to embedding layers dropout
        
        # Used for data collator
        self.tokenizer = BertTokenizer(
            vocab_file,
            do_lower_case=False,
            **special_tokens
        )
        self.model = self.get_model(family)

    def get_model(self, family):

        if family == "fttransf_flatten":
            model = FTTransformerFlatten(self.config, self.vocab)
        elif family == "tabbie":
            model = Tabbie(self.config, self.vocab)
        elif family == "column_tabbert":
            model = ColumnTabBERT(self.config, self.vocab)
        elif family == "row_tabbert":
            model = RowTabBERT(self.config, self.vocab)
        elif family == "fieldy":
            model = Fieldy(self.config, self.vocab)
            
        return model


class FTTransformerFlatten(PreTrainedModel):
    # Class attributes
    config_class = FTTransformerFlattenConfig
    base_model_prefix = "fttransf_flatten"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab

        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )

        if config.pos_emb:
            self.pos_embeddings = nn.Embedding(
                config.seq_len + 1,
                config.hidden_size,
                padding_idx=config.pad_token_id, 
                sparse=False
            )
        if config.col_emb:
            self.col_embeddings = nn.Embedding(
                config.ncols + 1,
                config.hidden_size,
                padding_idx=None, 
                sparse=False
            )

        # Embed [CLS] token for the Sequence Transformer (dim = dimension of the Seq Transformer)
        self.special_embeddings = nn.Embedding(
            len(self.vocab.token2id["SPECIAL"]),
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )
        self.sequence_transformer = CustomBertForMaskedLM(self.config, vocab)

    def setup_col_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.col_emb:
            column_ids = torch.arange(
                self.config.ncols 
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            column_ids = None
        return column_ids
    
    def setup_pos_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.pos_emb:
            position_ids = torch.arange(
                self.config.seq_len
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            position_ids = None
        return position_ids

    def forward(self, input_ids, **input_args):
        # 0. Setup variations
        position_ids = self.setup_pos_enc(input_ids)
        column_ids = self.setup_col_enc(input_ids)

        batch_size, nrows, ncols = input_ids.shape # [bs, nrows, ncols]
        # Embed all IDs
        inputs_embeds = self.word_embeddings(input_ids) # [bs, 1+nrows, 1+ncols, hs]

        inputs_embeds = inputs_embeds.flatten(start_dim=1, end_dim=2) # [bs, ncols * nrows, hs]
        
        # Insert the [CLS] token at the beggining for the Final Transformer
        cls_id = self.vocab.token2id["SPECIAL"]["[CLS]"][1]
        cls_id = torch.tensor([cls_id]).to(inputs_embeds.device)
        cls_embed = self.special_embeddings(cls_id)
        cls_embed = cls_embed.unsqueeze(dim=0).to(inputs_embeds.device)
        inputs_embeds = torch.cat(
            [
                cls_embed.repeat(batch_size, 1, 1), 
                inputs_embeds
            ], 
            dim=1
        ) # (bsz, 1 + seq_len * ncols, hidden_size)

        # Adapt the padding attention mask if needed (e.g. KDD data)
        if input_args["attention_mask"] is not None:
            final_mask = []
            for sample_id in range(input_args["attention_mask"].shape[0]):
                mask_cols = input_args["attention_mask"][sample_id].transpose(dim0=0, dim1=1) 
                sample_mask = mask_cols.flatten()
                sample_mask = torch.cat(
                        [
                            torch.tensor([1]).to(mask_cols.device), # [CLS]
                            sample_mask
                        ],
                        dim=0
                    )
                final_mask.append(sample_mask)
            
            input_args["attention_mask"] = torch.stack(final_mask) # [bsz, 1 + seq_len * ncols]

        # 2. Sequence Transformer
        if position_ids is not None and column_ids is not None:
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=torch.cat(
                    [   torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(position_ids.device),
                        torch.repeat_interleave(position_ids, self.config.ncols, dim=1),
                    ],
                    dim=1
                ),
                position_ids=torch.cat(
                    [   
                        torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(position_ids.device),
                        position_ids.repeat(1, self.config.ncols),
                    ],
                    dim=1
                ),
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)
        elif column_ids is not None:
            device = inputs_embeds.device
            column_ids = torch.arange(self.config.seq_len).unsqueeze(0).repeat(batch_size, 1).long().to(device)
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=None,
                position_ids=torch.cat(
                    [   
                        torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(device),
                        column_ids.repeat(1, self.config.ncols),
                    ],
                    dim=1
                ),
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)
        elif position_ids is not None:
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=torch.cat(
                    [   torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(position_ids.device),
                        torch.repeat_interleave(position_ids, self.config.ncols, dim=1),
                    ],
                    dim=1
                ),
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)
        else:
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)

        return seq_embed


class Tabbie(PreTrainedModel):
    # Class attributes
    config_class = TabbieConfig
    base_model_prefix = "tabbie"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab

        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )

        if config.pos_emb:
            self.pos_embeddings = nn.Embedding(
                config.seq_len + 1,
                config.hidden_size,
                padding_idx=config.pad_token_id, 
                sparse=False
            )
        if config.col_emb:
            self.col_embeddings = nn.Embedding(
                config.ncols + 1,
                config.hidden_size,
                padding_idx=None, 
                sparse=False
            )

        # Embed [CLS] token for the Sequence Transformer (dim = dimension of the Seq Transformer)
        self.special_embeddings = nn.Embedding(
            len(self.vocab.token2id["SPECIAL"]),
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )
        self.field_transformer_col = nn.ModuleDict({
            f"coltransf_layer_{i}": ColumnTabBERTEmbeddings(self.config, vocab, cls_col=True)
            for i
            in range(self.config.tabbiefieldtransf_nlayers)
        })
        self.field_transformer_row = nn.ModuleDict({
            f"rowtransf_layer_{i}": RowTabBERTEmbeddings(self.config, vocab, cls_row=True)
            for i
            in range(self.config.tabbiefieldtransf_nlayers)
        })
        
        self.mlm_head = TabFormerBertOnlyMLMHead(config)
        self.ft_head = nn.Linear(config.hidden_size* config.seq_len, config.num_ft_labels)

    def setup_col_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.col_emb:
            column_ids = torch.arange(
                self.config.ncols + 1 # If using CLSCOL and CLSROW
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            column_ids = None
        return column_ids
    
    def setup_pos_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.pos_emb:
            position_ids = torch.arange(
                self.config.seq_len + 1 # If using CLSCOL and CLSROW
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            position_ids = None
        return position_ids

    def forward(self, input_ids, **input_args):
        # 0. Setup variations
        position_ids = self.setup_pos_enc(input_ids)
        column_ids = self.setup_col_enc(input_ids)
        batch_size, nrows, ncols = input_ids.shape # [bs, nrows, ncols]

        # Add [CLS_COL] and [CLS_ROW]
        clsrow_id = self.vocab.token2id["SPECIAL"]["[BOS]"][1]
        clsrow_id = torch.tensor([clsrow_id]).to(input_ids.device)
        input_ids = torch.cat(
            [
                clsrow_id.repeat(batch_size, self.config.seq_len, 1), 
                input_ids
            ], 
            dim=2
        ) # (bsz, seq_len, 1+ncols)
        clscol_id = self.vocab.token2id["SPECIAL"]["[EOS]"][1]
        clscol_id = torch.tensor([clscol_id]).to(input_ids.device)
        input_ids = torch.cat(
            [
                clscol_id.repeat(batch_size, 1, self.config.ncols + 1), 
                input_ids
            ], 
            dim=1
        ) # (bsz, 1+seq_len, 1+ncols)
        
        # Embed all IDs
        inputs_embeds = self.word_embeddings(input_ids) # [bs, 1+nrows, 1+ncols, hs]

        # Add pos emb.
        if position_ids is not None :
            pos_embeds = self.pos_embeddings(position_ids).unsqueeze(1)
            pos_embeds = pos_embeds.repeat(1, self.config.ncols + 1, 1, 1)
            inputs_embeds = inputs_embeds + pos_embeds.permute(0, 2, 1, 3)

        # Add col emb.
        if column_ids is not None :
            col_embeds = self.col_embeddings(column_ids).unsqueeze(1)
            inputs_embeds = inputs_embeds + col_embeds

        for i in range(self.config.tabbiefieldtransf_nlayers):
            
            # 1.1 Field Transformer (by column)
            if input_args["attention_mask"] is not None: # KDD
                # Padding attention mask in PyTorch transformer format
                # The attention mask is in "row-first" format
                mask = input_args["attention_mask"].float()
                mask = mask.transpose(dim0=1, dim1=2)
                mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                mask = torch.cat([
                    torch.zeros(batch_size, self.config.seq_len, 1).to(mask.device), 
                    mask
                ], dim=2)  # CLS row is ttended
                mask = torch.cat([
                    torch.zeros(batch_size, 1, self.config.ncols + 1).to(mask.device), 
                    mask
                ], dim=1) # CLS col is attended
                src_key_padding_mask = mask
            else:
                src_key_padding_mask = None
            inputs_embeds_col = self.field_transformer_col[f"coltransf_layer_{i}"](
                input_ids=inputs_embeds.transpose(dim0=1, dim1=2), 
                src_key_padding_mask=src_key_padding_mask,
                position_ids=None,
                skip_word_emb=True,
                cls_col=True, # If using CLROW and CLSCOL
            ) # [bsz, ncols, hidden_size]

            # 1.2 Field Transformer (by row)
            inputs_embeds_row = self.field_transformer_row[f"rowtransf_layer_{i}"](
                input_ids=inputs_embeds, 
                column_ids=None,
                skip_word_emb=True,
                cls_row=True, # If using CLROW and CLSCOL
            ) # [bsz, seq_len, hidden_size]

            
            # Between each layer: Compute avg(row, col) to create the cells embeddings
            inputs_embeds = (inputs_embeds_col.unsqueeze(2) + inputs_embeds_row.unsqueeze(1)) # [bs, 1+ncols, 1+nrows, hs]
            inputs_embeds = inputs_embeds / 2.0 # [bs, 1+ncols, 1+nrows, hs]
            inputs_embeds = inputs_embeds.permute(0, 2, 1, 3) # If using CLROW and CLSCOL, # [bs, 1+nrows, 1+ncols, hs]
    
     
        # 2. FFN

        # Fine-tuning mode
        if input_args.get('masked_lm_labels') is None:
            inputs_embeds = inputs_embeds[:, 1:, 0, :] # Take only the CLSROWs
            inputs_embeds = inputs_embeds.flatten(start_dim=1) # Take only the CLSROWs
            prediction_scores = self.ft_head(inputs_embeds) 
            return (prediction_scores, None)
        
        # Pre-training MLM mode
        else: 
            # if using CLS, Discard the [CLS] token
            inputs_embeds = inputs_embeds[:, 1:, 1:, :]
            inputs_embeds = inputs_embeds.reshape(batch_size, self.config.ncols * self.config.seq_len, -1)
            prediction_scores = self.mlm_head(inputs_embeds)
            if self.config.mlm_loss == "all":
                # Assess MLM prediction on masked tokens, across all vocab logits distribution
                criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore unmasked tokens
                total_masked_lm_loss = criterion(
                    prediction_scores.reshape(-1, len(self.vocab)),
                    input_args['masked_lm_labels'].reshape(-1)
                )
            return (total_masked_lm_loss, None, prediction_scores)
        

class ColumnTabBERT(PreTrainedModel):
    # Class attributes
    config_class = ColumnTabBERTConfig
    base_model_prefix = "column_tabbert"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.field_transformer = ColumnTabBERTEmbeddings(self.config, vocab)
        self.sequence_transformer = CustomBertForMaskedLM(self.config, vocab)

    def setup_col_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.col_emb:
            column_ids = torch.arange(
                self.config.ncols + 1 # Accounts for [CLS] 
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            column_ids = None
        return column_ids
    
    def setup_pos_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.pos_emb:
            position_ids = torch.arange(
                self.config.seq_len
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            position_ids = None
        return position_ids

    def forward(self, input_ids, **input_args):
        batch_size, seq_len, ncols = input_ids.shape
        
        # 0. Setup variations
        position_ids = self.setup_pos_enc(input_ids)
        column_ids = self.setup_col_enc(input_ids)

        # 1. Field Transformer (by column)
        if input_args["attention_mask"] is not None: # KDD
            # Padding attention mask in PyTorch transformer format
            mask = input_args["attention_mask"].float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None

        inputs_embeds = self.field_transformer(
            input_ids=input_ids, 
            position_ids=position_ids, 
            src_key_padding_mask=src_key_padding_mask,
        ) # [bsz, ncols, hidden_size]

        if self.config.insert_sep:
            # Insert [SEP] tokens between each row for the Sequence Transformer
            sep_id = self.vocab.token2id["SPECIAL"]["[SEP]"][1]
            sep_id = torch.tensor([sep_id]).to(inputs_embeds.device)
            sep_embed = self.field_transformer.word_embeddings(sep_id)
            sep_embed = sep_embed.unsqueeze(dim=0).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.repeat_interleave(2, dim=1)
            inputs_embeds[:, 1::2, :] = sep_embed

        # Insert the [CLS] token at the beggining for the Sequence Transformer
        cls_id = self.vocab.token2id["SPECIAL"]["[CLS]"][1]
        cls_id = torch.tensor([cls_id]).to(inputs_embeds.device)
        cls_embed = self.field_transformer.special_embeddings(cls_id)
        cls_embed = cls_embed.unsqueeze(dim=0).to(inputs_embeds.device)
        inputs_embeds = torch.cat(
            [
                cls_embed.repeat(batch_size, 1, 1), 
                inputs_embeds
            ], 
            dim=1
        ) # (bsz, ncols + 1, hidden_size)

        # No padding attention mask between Columns, it has been done in the Field Transf. (for KDD data)
        input_args["attention_mask"] = None

        # 2. Sequence Transformer
        seq_embed = self.sequence_transformer(
            input_ids=None,
            token_type_ids=None,
            position_ids=column_ids,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            **input_args
        ) # (bsz, ncols + 1, hidden_size)

        return seq_embed
    

class RowTabBERT(PreTrainedModel):
    # Class attributes
    config_class = RowTabBERTConfig
    base_model_prefix = "row_tabbert"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.field_transformer = RowTabBERTEmbeddings(self.config, vocab)
        self.sequence_transformer = CustomBertForMaskedLM(self.config, vocab)

    def setup_col_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.col_emb:
            column_ids = torch.arange(
                self.config.ncols
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            column_ids = None
        return column_ids
    
    def setup_pos_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.pos_emb:
            position_ids = torch.arange(
                self.config.seq_len + 1 # accounts for [CLS] tokens
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            position_ids = None
        return position_ids

    def forward(self, input_ids, **input_args):
        batch_size, seq_len, ncols = input_ids.shape
        
        # 0. Setup variations
        position_ids = self.setup_pos_enc(input_ids)
        column_ids = self.setup_col_enc(input_ids)

        # 1. Field Transformer
        inputs_embeds = self.field_transformer(
            input_ids=input_ids, 
            column_ids=column_ids, 
        ) # [bsz, seq_len, hidden_size]

        # Insert the [CLS] token at the beggining for the Sequence Transformer
        cls_id = self.vocab.token2id["SPECIAL"]["[CLS]"][1]
        cls_id = torch.tensor([cls_id]).to(inputs_embeds.device)
        cls_embed = self.field_transformer.special_embeddings(cls_id)
        cls_embed = cls_embed.unsqueeze(dim=0).to(inputs_embeds.device)
        inputs_embeds = torch.cat(
            [
                cls_embed.repeat(batch_size, 1, 1), 
                inputs_embeds
            ], 
            dim=1
        ) # (bsz, seq_len + 1, hidden_size)

        # Adapt the padding attention mask if needed (e.g. KDD data)
        if input_args["attention_mask"] is not None:
            mask = input_args["attention_mask"]
            mask = mask[:, :, 0] # If one token in a row is masked, we mask the full row
            if self.config.insert_sep:
                mask = mask.repeat_interleave(2, dim=1) # [SEP]
            mask = torch.cat(
                [
                    torch.ones(mask.shape[0], 1).to(mask.device), # [CLS]
                    mask
                ],
                dim=1
            )
            input_args["attention_mask"] = mask

        # 2. Sequence Transformer
        seq_embed = self.sequence_transformer(
            input_ids=None,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            **input_args
        ) # (bsz, seq_len + 1, hidden_size)

        return seq_embed


class Fieldy(PreTrainedModel):
    # Class attributes
    config_class = FieldyConfig
    base_model_prefix = "fieldy"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab

        # Embed [CLS] token for the Sequence Transformer (dim = dimension of the Seq Transformer)
        self.special_embeddings = nn.Embedding(
            len(self.vocab.token2id["SPECIAL"]),
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )
        self.field_transformer_col = ColumnTabBERTEmbeddings(self.config, vocab)
        self.field_transformer_row = RowTabBERTEmbeddings(self.config, vocab)
        self.linear = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.sequence_transformer = CustomBertForMaskedLM(self.config, vocab)

    def setup_col_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.col_emb:
            column_ids = torch.arange(
                self.config.ncols 
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            column_ids = None
        return column_ids
    
    def setup_pos_enc(self, input_ids):
        batch_size, seq_len, ncols = input_ids.shape
        device = input_ids.device
        if self.config.pos_emb:
            position_ids = torch.arange(
                self.config.seq_len
            ).unsqueeze(0).repeat(batch_size, 1).long().to(device)
        else:
            position_ids = None
        return position_ids

    def forward(self, input_ids, **input_args):
        batch_size, seq_len, ncols = input_ids.shape
        
        # 0. Setup variations
        position_ids = self.setup_pos_enc(input_ids)
        column_ids = self.setup_col_enc(input_ids)

        # 1.1 Field Transformer (by column)
        if input_args["attention_mask"] is not None: # KDD
            # Padding attention mask in PyTorch transformer format
            # The attention mask is in "row-first" format
            mask = input_args["attention_mask"].float().transpose(dim0=1, dim1=2)
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None
        inputs_embeds_col = self.field_transformer_col(
            input_ids=input_ids.transpose(dim0=1, dim1=2), 
            position_ids=None, 
            src_key_padding_mask=src_key_padding_mask,
        ) # [bsz, ncols, hidden_size]

        # 1.2 Field Transformer (by row)
        inputs_embeds_row = self.field_transformer_row(
            input_ids=input_ids, 
            column_ids=None, 
        ) # [bsz, seq_len, hidden_size]

        # 1.3 Compute Linear(row, col) to create the cells embeddings
        inputs_embeds_col = inputs_embeds_col.unsqueeze(2).repeat(1, 1, self.config.seq_len, 1)
        inputs_embeds_row = inputs_embeds_row.unsqueeze(1).repeat(1, self.config.ncols, 1, 1)
        inputs_embeds = torch.cat([inputs_embeds_col, inputs_embeds_row], dim=3) # [bs, ncols, nrows, hs + hs]
        inputs_embeds = inputs_embeds.flatten(start_dim=1, end_dim=2) # [bs, ncols * nrows, hs + hs]
        inputs_embeds = self.linear(inputs_embeds) # [bs, ncols * nrows, hs]
        
        # Insert the [CLS] token at the beggining for the Final Transformer
        cls_id = self.vocab.token2id["SPECIAL"]["[CLS]"][1]
        cls_id = torch.tensor([cls_id]).to(inputs_embeds.device)
        cls_embed = self.special_embeddings(cls_id)
        cls_embed = cls_embed.unsqueeze(dim=0).to(inputs_embeds.device)
        inputs_embeds = torch.cat(
            [
                cls_embed.repeat(batch_size, 1, 1), 
                inputs_embeds
            ], 
            dim=1
        ) # (bsz, 1 + seq_len * ncols, hidden_size)

        # Adapt the padding attention mask if needed (e.g. KDD data)
        if input_args["attention_mask"] is not None:
            final_mask = []
            for sample_id in range(input_args["attention_mask"].shape[0]):
                # The attention mask is given in "row-first" format, but we compute AVG(row, col) starting with col, then rows to produce the final sequence
                mask_cols = input_args["attention_mask"][sample_id].transpose(dim0=0, dim1=1) 
                sample_mask = mask_cols.flatten()
                # Attend to [CLS]
                sample_mask = torch.cat(
                        [
                            torch.tensor([1]).to(mask.device), # [CLS]
                            sample_mask
                        ],
                        dim=0
                    )
                final_mask.append(sample_mask) 
            input_args["attention_mask"] = torch.stack(final_mask) # [bsz, 1 + seq_len * ncols]

        # 2. Sequence Transformer
        if position_ids is not None and column_ids is not None:
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=torch.cat(
                    [   torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(position_ids.device),
                        torch.repeat_interleave(position_ids, self.config.ncols, dim=1),
                    ],
                    dim=1
                ),
                position_ids=torch.cat(
                    [   
                        torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(position_ids.device),
                        position_ids.repeat(1, self.config.ncols),
                    ],
                    dim=1
                ),
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)
        elif column_ids is not None:
            device = inputs_embeds.device
            column_ids = torch.arange(self.config.seq_len).unsqueeze(0).repeat(batch_size, 1).long().to(device)
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=None,
                position_ids=torch.cat(
                    [   
                        torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(device),
                        column_ids.repeat(1, self.config.ncols),
                    ],
                    dim=1
                ),
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)
        elif position_ids is not None:
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=torch.cat(
                    [   torch.tensor([self.config.seq_len + 1]).unsqueeze(1).repeat(batch_size, 1).to(position_ids.device),
                        torch.repeat_interleave(position_ids, self.config.ncols, dim=1),
                    ],
                    dim=1
                ),
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)
        else:
            seq_embed = self.sequence_transformer(
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                **input_args
            ) # During pre-training: (loss, full_outputs, prediction_scores)

        return seq_embed
