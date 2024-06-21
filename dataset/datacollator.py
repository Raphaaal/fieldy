from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
from transformers import DataCollatorForLanguageModeling
from transformers import DefaultDataCollator
from torch.nn.utils.rnn import pad_sequence

from misc.utils import seed_everything
import random


class RowTabBERTDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(
            self, 
            ncols: int = None,
            seq_len: int = None,
            data_type: str = None,
            seed: int = None,
            randomize_seq: bool = True,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.ncols = ncols
        self.seq_len = seq_len
        self.is_ncols_updated = False
        self.ncols_initial = self.ncols
        self.data_type = data_type
        self.randomize_seq = randomize_seq
        if seed:
            seed_everything(seed)

    def _constant_pad_sequence(
        self,
        examples, 
        batch_first=True, 
        padding_value=None,
        randomize_seq=True,
    ):
        # Select a random sequence of consecutive transactions for each sample
        randomized_examples = []
        for i, ex in enumerate(examples):
            if randomize_seq:
                # Pre-training on KDD
                if ex.shape[0] < self.seq_len:
                    ix = 0
                else:
                    ix = random.randint(0, ex.shape[0] - self.seq_len)
                randomized_ex = ex[ix: ix + self.seq_len, :]
            else: # Fine-tuning on KDD
                if ex.shape[0] < self.seq_len:
                    ix = 0
                else:
                    ix = ex.shape[0] - self.seq_len
                randomized_ex = ex[ix: ix + self.seq_len, :] # During fine-tuning, we take the last consecutive transactions in KDD
            if i == 0:
                # pad first sequence to desired length
                padder = torch.nn.ConstantPad2d(
                    (0, 0, 0, self.seq_len - randomized_ex.shape[0]),
                    padding_value
                )
                randomized_ex = padder(randomized_ex)
            randomized_examples.append(randomized_ex)
        
        # pad all the remaining randomized sequences
        padded_examples = pad_sequence(
            randomized_examples, 
            batch_first=batch_first, 
            padding_value=padding_value
        )
        return padded_examples
    
    def _tensorize_batch(self, examples: List[torch.Tensor], randomize_seq=True) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if (
            are_tensors_same_length and 
            not (len(examples) == 1  and self.data_type == 'kdd') # Corner case for KDD fine-tuning -> One batch may be made of a single example
        ):
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return self._constant_pad_sequence(
                examples=examples, 
                batch_first=True, 
                padding_value=self.tokenizer.pad_token_id,
                randomize_seq=randomize_seq,
            )

    def __call__(
            self, 
            examples: List[
                Union[List[int], 
                torch.Tensor, 
                Dict[str, torch.Tensor]]
            ],
    ) -> Dict[str, torch.Tensor]:
        
        # Don't care about the target for MLM
        batch = self._tensorize_batch(
            [
                ex[0] 
                for ex 
                in examples
            ],
            randomize_seq=self.randomize_seq, # For KDD fine-tuning
        )
        sz = batch.shape
        sz, batch, timedeltas = self.extract_timedeltas(
            batch, 
            end_shape=sz, 
            flatten=False,
        )
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels, padding_attention_mask = self.mask_tokens(batch) 
            return {
                "input_ids": inputs.view(sz), 
                "masked_lm_labels": labels.view(sz),
                "timedeltas": timedeltas,
                "attention_mask": padding_attention_mask.view(sz) if self.data_type == "kdd" else None,
            }
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": batch, 
                "labels": labels
            }
        
    def extract_timedeltas(
            self, 
            batch: torch.Tensor,
            end_shape: list,
            flatten: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the timedeltas column that will be used for positional encoding.
        Removes the colum from the input.
        """

        timedeltas = None

        return (end_shape, batch, timedeltas)
    
    def mask_tokens(
            self, 
            inputs: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Taken from: https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/data/data_collator.py#L751
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary "
                "for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training 
        # (with probability args.mlm_probability, defaults to 0.15 in BERT)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Do NOT mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, 
                already_has_special_tokens=True
            ) 
            for val 
            in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(
                special_tokens_mask, 
                dtype=torch.bool
            ), 
            value=0.0
        )

        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Select the initial [MASK] indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Attention mask if needed (e.g., because of padding)
        # probability be `1` (masked), however in albert model attention mask `0` means masked, revert the value
        attention_mask = torch.ones_like(labels)
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=0.)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            7, # Do not replace by one of the 7 special tokens
            len(self.tokenizer), 
            labels.shape, 
            dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (i.e., the remaining 10% of the time) 
        # we keep the masked input tokens unchanged

        return inputs, labels, attention_mask
    

class ColumnTabBERTDataCollatorForLanguageModeling(RowTabBERTDataCollatorForLanguageModeling):

    def __call__(
            self, 
            examples: List[
                Union[List[int], 
                torch.Tensor, 
                Dict[str, torch.Tensor]]
            ],
    ) -> Dict[str, torch.Tensor]:
        
        # Don't care about the target for MLM
        batch = self._tensorize_batch(
            [
                ex[0] 
                for ex 
                in examples
            ],
            randomize_seq=self.randomize_seq, # For KDD fine-tuning
        )
        sz = batch.shape
        
        if self.mlm:
            batch = batch.view(sz[0], -1)
            sz, batch, timedeltas = self.extract_timedeltas(
                batch, 
                end_shape=sz, 
                flatten=True
            )
            inputs, labels, padding_attention_mask = self.mask_tokens(batch) 
            return {
                "input_ids": inputs.view(sz).transpose(dim0=1, dim1=2), 
                "masked_lm_labels": labels.view(sz).transpose(dim0=1, dim1=2).flatten(start_dim=1), # New used for KDD results
                "timedeltas": timedeltas,
                "attention_mask": padding_attention_mask.view(sz).transpose(dim0=1, dim1=2) if self.data_type == "kdd" else None,
            }
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": batch, 
                "labels": labels,
            }
        

class FieldyDataCollatorForLanguageModeling(RowTabBERTDataCollatorForLanguageModeling):
        
    def __call__(
            self, 
            examples: List[
                Union[List[int], 
                torch.Tensor, 
                Dict[str, torch.Tensor]]
            ],
    ) -> Dict[str, torch.Tensor]:
        
        # Don't care about the target for MLM
        batch = self._tensorize_batch(
            [
                ex[0] 
                for ex 
                in examples
            ],
            randomize_seq=self.randomize_seq
        )
        sz = batch.shape
        sz, batch, timedeltas = self.extract_timedeltas(
            batch, 
            end_shape=sz, 
            flatten=False,
        )
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels, padding_attention_mask = self.mask_tokens(batch) 
            return {
                "input_ids": inputs.view(sz), # Return organized by row, will be handled in the model forward()
                "masked_lm_labels": labels.view(sz).transpose(dim0=1, dim1=2).flatten(start_dim=1), # Return labels flattened column by column (this is the order in which avg(col,row) is computed), NEW used for kdd
                "timedeltas": timedeltas,
                "attention_mask": padding_attention_mask.view(sz) if self.data_type == "kdd" else None,
            }
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": batch, 
                "labels": labels
            }
    

class FTTransformerFlattenDataCollatorForLanguageModeling(RowTabBERTDataCollatorForLanguageModeling):
    pass


class TabbieDataCollatorForLanguageModeling(RowTabBERTDataCollatorForLanguageModeling):
    pass


class RowTabBERTDataCollatorForFineTuning(RowTabBERTDataCollatorForLanguageModeling):

    def __call__(
            self, 
            examples: List[Union[List[int], 
            torch.Tensor, 
            Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        inputs = self._tensorize_batch(
            [
                ex[0] 
                for ex 
                in examples
            ], 
            randomize_seq=False, # For KDD fine-tuning
        )
        sz = inputs.shape
        sz, inputs, timedeltas = self.extract_timedeltas(
            inputs, 
            end_shape=sz, 
            flatten=False
        )
        if self.data_type == "kdd":
            # No need to tensorize batch with padding
            labels = torch.tensor([
                ex[1] 
                for ex 
                in examples
            ]).to(examples[0][1].device)
        elif self.data_type == "card":
            # No need to tensorize batch
            labels = torch.tensor([
                1 
                if 1 
                in ex[1] 
                else 0 
                for ex 
                in examples
            ])
        elif self.data_type == "prsa":
            labels = self._tensorize_batch([
                ex[1] 
                for ex 
                in examples
            ])

        # Special case when padding (e.g., for KDD)
        attention_mask = torch.ones_like(inputs)
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=0.)

        return {
            "input_ids": inputs, 
            "labels": labels,
            "timedeltas": timedeltas,
            "attention_mask": attention_mask if self.data_type == "kdd" else None,
        }
    

class ColumnTabBERTDataCollatorForFineTuning(RowTabBERTDataCollatorForLanguageModeling):

    def __call__(
            self, 
            examples: List[Union[List[int], 
            torch.Tensor, 
            Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        inputs = self._tensorize_batch([
                ex[0] 
                for ex 
                in examples
            ],
            randomize_seq=False, # For KDD fine-tuning
        )
        inputs = inputs.transpose(dim0=1, dim1=2)
        sz = inputs.shape
        sz, inputs, timedeltas = self.extract_timedeltas(
            inputs, 
            end_shape=sz, 
            flatten=False
        )
        if self.data_type == "kdd":
            # No need to tensorize batch with padding
            labels = torch.tensor([
                ex[1] 
                for ex 
                in examples
            ]).to(examples[0][1].device)
        elif self.data_type == "card":
            # No need to tensorize batch
            labels = torch.tensor([
                1 
                if 1 
                in ex[1] 
                else 0 
                for ex 
                in examples
            ])
        elif self.data_type == "prsa":
            labels = self._tensorize_batch([
                ex[1] 
                for ex 
                in examples
            ])

        # Special case when padding (e.g., for KDD)
        attention_mask = torch.ones_like(inputs)
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=0.)

        return {
            "input_ids": inputs, 
            "labels": labels,
            "timedeltas": timedeltas,
            "attention_mask": attention_mask if self.data_type == "kdd" else None,
        }
    

class FieldyDataCollatorForFineTuning(RowTabBERTDataCollatorForFineTuning):
    pass


class FTTransformerFlattenDataCollatorForFineTuning(RowTabBERTDataCollatorForFineTuning):
    pass


class TabbieDataCollatorForFineTuning(RowTabBERTDataCollatorForFineTuning):
    pass