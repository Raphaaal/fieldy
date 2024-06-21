from typing import (
    Optional,
    Union,
    Tuple,
    List,
)

import torch.nn as nn

from transformers.activations import ACT2FN 
from transformers import (
    BertForMaskedLM,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder, 
    BertPooler, 
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertPreTrainedModel,
)

import torch


class RowTabBERTEmbeddings(nn.Module):

    def __init__(self, config, vocab, cls_row=False):
        super().__init__()
        self.config = config
        self.vocab = vocab

        if self.config.family in ["row_tabbert", "fieldy"]:
            self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )

            if self.config.family == "row_tabbert":
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size, 
            nhead=self.config.fieldtransf_nheads,
            dim_feedforward=config.hidden_size,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config.fieldtransf_nlayers
        )
        
        if cls_row:
            self.lin_proj = nn.Linear(
                config.hidden_size * (config.ncols + 1), 
                config.hidden_size
            )
        else:
            self.lin_proj = nn.Linear(
                config.hidden_size * config.ncols, 
                config.hidden_size
            )

    def forward(self, input_ids, column_ids=None, skip_word_emb=False, cls_row=False):
        # Tokens embedding
        if skip_word_emb:
            inputs_embeds = input_ids
        else:
            inputs_embeds = self.word_embeddings(input_ids) # (bsz, seq_len, nfeatures, embed_dim)

        # Embed column index
        if column_ids is not None:            
            col_embeds = self.col_embeddings(column_ids).unsqueeze(1)
            if cls_row:
                col_embeds = col_embeds.repeat(1, self.config.seq_len + 1, 1, 1)
            else:
                col_embeds = col_embeds.repeat(1, self.config.seq_len, 1, 1)
            inputs_embeds = inputs_embeds + col_embeds

        # Attention between fields
        embeds_shape = list(inputs_embeds.shape)
        inputs_embeds = inputs_embeds.reshape(-1, embeds_shape[2], embeds_shape[3]) # Reshape for attending between fields
        inputs_embeds = self.transformer_encoder(inputs_embeds)

        # Reshape into original rows
        if cls_row:
            inputs_embeds = inputs_embeds.reshape(
                embeds_shape[0], 
                self.config.seq_len+1, 
                self.config.hidden_size * (self.config.ncols +1)
            )
        else:
            inputs_embeds = inputs_embeds.reshape(
                embeds_shape[0], 
                embeds_shape[1], 
                self.config.hidden_size * self.config.ncols
            ) 

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds


class ColumnTabBERTEmbeddings(nn.Module):

    def __init__(self, config, vocab, cls_col=False):
        super().__init__()
        self.config = config
        self.vocab = vocab

        if self.config.family in ["column_tabbert", "fieldy"]:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, 
                config.hidden_size,
                padding_idx=getattr(config, 'pad_token_id', 0), 
                sparse=False
            )

            if self.config.family == "column_tabbert":
                if self.config.pos_emb:
                    self.pos_embeddings = nn.Embedding(
                        config.seq_len + 1,
                        config.hidden_size,
                        padding_idx=config.pad_token_id, 
                        sparse=False
                    )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size, 
            nhead=self.config.fieldtransf_nheads,
            dim_feedforward=config.hidden_size,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config.fieldtransf_nlayers
        )
        
        if cls_col:
            self.lin_proj = nn.Linear(
                config.hidden_size * (config.seq_len + 1), 
                config.hidden_size
            )
        else:
            self.lin_proj = nn.Linear(
                config.hidden_size * config.seq_len, 
                config.hidden_size
            )        

        # Embed [CLS] token for the Sequence Transformer (dim = dimension of the Seq Transformer)
        self.special_embeddings = nn.Embedding(
            len(self.vocab.token2id["SPECIAL"]), 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0), 
            sparse=False
        )

    def forward(self, input_ids, position_ids=None, src_key_padding_mask=None, skip_word_emb=False, cls_col=False):
        # Tokens embedding
        if skip_word_emb:
            inputs_embeds = input_ids
        else:
            inputs_embeds = self.word_embeddings(input_ids) # (bsz, seq_len, nfeatures, embed_dim)

        # Embed column index
        if position_ids is not None:            
            pos_embeds = self.pos_embeddings(position_ids).unsqueeze(1)
            if cls_col:
                pos_embeds = pos_embeds.repeat(1, self.config.ncols + 1, 1, 1)
            else:
                pos_embeds = pos_embeds.repeat(1, self.config.ncols, 1, 1)
            inputs_embeds = inputs_embeds + pos_embeds

        # Attention between fields
        embeds_shape = list(inputs_embeds.shape)
        inputs_embeds = inputs_embeds.reshape(-1, embeds_shape[2], embeds_shape[3])
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.reshape(-1, embeds_shape[2]) # [bs * ncols, seqlen]
        inputs_embeds = self.transformer_encoder(
            inputs_embeds, 
            src_key_padding_mask=src_key_padding_mask
        )

        # Reshape into original columns
        if cls_col:
            inputs_embeds = inputs_embeds.reshape(
                embeds_shape[0], 
                self.config.ncols + 1, 
                self.config.hidden_size * (self.config.seq_len+1) 
            )
        else:
            inputs_embeds = inputs_embeds.reshape(
                embeds_shape[0], 
                embeds_shape[1], 
                self.config.hidden_size * self.config.seq_len
            ) 
        
        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds


class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.family in ["fttransf_flatten", "tabbie", "fieldy"]:
            self.dense = nn.Linear(
                config.hidden_size, 
                config.hidden_size
            )
        elif config.family == "column_tabbert":
            # When recovering fields from columns for MLM, hidden_size hase been reshaped (i.e., divided by seqlen)
            self.dense = nn.Linear(
                int(config.hidden_size / config.seq_len), 
                config.hidden_size
            )
        elif config.family == "row_tabbert": 
            # When recovering fields from rows for MLM, hidden_size hase been reshaped (i.e., divided by ncols)
            self.dense = nn.Linear(
                int(config.hidden_size / config.ncols), 
                config.hidden_size
            )
       
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    

class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class CustomBertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.family in ["fttransf_flatten", "fieldy"]:
            self.token_type_embeddings = nn.Embedding(config.seq_len + 2, config.hidden_size)
        else:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Token type (done before in the field transformer for TabBERT)
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds

        # Position (optional)
        if position_ids is not None:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class CustomBertForMaskedLM(BertForMaskedLM):

    def __init__(self, config, vocab):
        super().__init__(config)
        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.ft_head = nn.Linear(config.hidden_size, config.num_ft_labels)
        self.init_weights()
        self.config = config
        self.bert = CustomBertModel(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            labels=None,
            timedeltas=None,
            **kwargs,
    ):
        outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

        # Fine-tuning mode
        if not torch.is_tensor(masked_lm_labels):
            cls_token_embeds = outputs.pooler_output  # [bsz, hidden_dim]
            # Add Dropout?
            prediction_scores = self.ft_head(cls_token_embeds) 
            # return prediction_scores
            return (prediction_scores, outputs)
        
        # Pre-training MLM mode
        else: 
            # Discard the [CLS] token
            sequence_output = outputs.last_hidden_state
            sequence_output = sequence_output[:, 1:, :]

            if self.config.family == "fttransf_flatten":
                pass
            elif self.config.family == "tabbie":
                pass 
            elif self.config.family == "column_tabbert":
                # Recover field-level embeddings from the columns embeddings
                output_sz = list(sequence_output.shape) # [bsz, ncols, hidden_dim]
                expected_sz = [output_sz[0], output_sz[1] * self.config.seq_len, -1] # [bsz, nfields * ncols, hidden_dim / seqlen]
                sequence_output = sequence_output.reshape(expected_sz) 
                masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1) # [bsz, nfields * seqlen]
            elif self.config.family == "row_tabbert":
                # Recover field-level embeddings from the rows embeddings
                output_sz = list(sequence_output.shape) # [bsz, seqlen, hidden_dim]
                expected_sz = [output_sz[0], output_sz[1] * self.config.ncols, -1] # [bsz, nfields * seqlen, hidden_dim / nfields]
                sequence_output = sequence_output.reshape(expected_sz) 
                masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1) # [bsz, nfields * seqlen]
            elif self.config.family == "fieldy":
                pass 

            # MLM loss computation
            prediction_scores = self.cls(sequence_output) # [bsz, nfields * seqlen, vocab_sz]
            
            if self.config.mlm_loss == "all":
                # Assess MLM prediction on masked tokens, across all vocab logits distribution
                criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore unmasked tokens
                total_masked_lm_loss = criterion(
                    prediction_scores.reshape(-1, len(self.vocab)),
                    masked_lm_labels.reshape(-1)
                )

            elif self.config.mlm_loss == "attribute":
                # Assess MLM prediction on masked tokens field by field, among its possible 50 values
                total_masked_lm_loss = 0.
                seq_len = prediction_scores.size(1)
                field_names = self.vocab.get_field_keys(
                    remove_target=True, 
                    ignore_special=True,
                    remove_timedelta=self.config.timedelta,
                )

                for field_idx, field_name in enumerate(field_names):
                    col_ids = list(range(field_idx, seq_len, len(field_names)))
                    global_ids_field = self.vocab.get_field_ids(field_name)
                    prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
                    masked_lm_labels_field = masked_lm_labels[:, col_ids]
                    masked_lm_labels_field_local = self.vocab.get_from_global_ids(
                        global_ids=masked_lm_labels_field,
                        what_to_get='local_ids'
                    )
                    nfeas = len(global_ids_field)
                    loss_fct = self.get_criterion(
                        field_name, 
                        nfeas, 
                        prediction_scores.device
                    )

                    # Corner-case when the labels are full of -100 (i.e. no [MASK])
                    unique_labels = torch.unique(masked_lm_labels_field_local)
                    if unique_labels.shape[0] == 1 and unique_labels.item() == -100.:
                        #  Set loss to 0.
                        masked_lm_loss_field = torch.tensor(0.).to(
                            masked_lm_labels_field_local.device
                        )
                    else:
                        masked_lm_loss_field = loss_fct(
                            prediction_scores_field.view(-1, len(global_ids_field)),
                            masked_lm_labels_field_local.view(-1)
                        )

                    total_masked_lm_loss += masked_lm_loss_field

            return (total_masked_lm_loss, outputs, prediction_scores)

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):
        return nn.CrossEntropyLoss()


class CustomBertModel(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.config.attention_probs_dropout_prob = self.config.dropout
        self.embeddings = CustomBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )