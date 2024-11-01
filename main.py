from os import makedirs
from os.path import join
from pathlib import Path

import wandb
wandb.init(mode="disabled")

import sys           
import json
from collections import OrderedDict
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from args import define_main_parser
import pickle
import random

import torch
from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TrainerCallback
from transformers import TrainerState
from transformers import TrainerState
from transformers import TrainerControl

from dataset.prsa import PRSADataset

from models.models import Model
from models.models import FTTransformerFlatten
from models.models import Tabbie
from models.models import ColumnTabBERT
from models.models import RowTabBERT
from models.models import Fieldy

from models.configs import FTTransformerFlattenConfig
from models.configs import TabbieConfig
from models.configs import ColumnTabBERTConfig
from models.configs import RowTabBERTConfig
from models.configs import FieldyConfig

from misc.utils import random_split_dataset
from misc.utils import combined_mape
from misc.utils import combined_rmse
from misc.utils import combined_rmse_torch
from misc.utils import seed_everything
from misc.utils import count_parameters
from misc.utils import count_parameters_field_transf
from misc.utils import count_parameters_seq_transf

from dataset.datacollator import FTTransformerFlattenDataCollatorForLanguageModeling
from dataset.datacollator import TabbieDataCollatorForLanguageModeling
from dataset.datacollator import ColumnTabBERTDataCollatorForLanguageModeling
from dataset.datacollator import RowTabBERTDataCollatorForLanguageModeling
from dataset.datacollator import FieldyDataCollatorForLanguageModeling

from dataset.datacollator import FTTransformerFlattenDataCollatorForFineTuning
from dataset.datacollator import TabbieDataCollatorForFineTuning
from dataset.datacollator import ColumnTabBERTDataCollatorForFineTuning
from dataset.datacollator import RowTabBERTDataCollatorForFineTuning
from dataset.datacollator import FieldyDataCollatorForFineTuning

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from pathlib import Path
import pandas as pd


class InspectCallback(TrainerCallback):
    """
    Inspect the same sample throughout training 
    i.e., displays preds and labels for comparison
    """

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.pop('dataset')
        self.args = kwargs.pop('args')
        self.log = kwargs.pop('logger')
        self.data_collator = kwargs.pop('data_collator')
        self.nexamples = kwargs.pop('nexamples')
        self.scaler = kwargs.pop('scaler')
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Gather the examples to inspect
        examples = []
        for i, (inpt, label) in enumerate(self.dataset):
            if i < self.nexamples:
                examples.append([inpt, label])
        model = kwargs["model"]
        examples = self.data_collator(examples)

        # Place them on the model's device
        examples_device = {}
        for k, v in examples.items():
            if isinstance(v, torch.Tensor):
                examples_device[k] = v.to(model.device)
            else:
                examples_device[k] = v

        # Run predictions on examples
        labels = examples_device["labels"]
        with torch.no_grad():
            # preds = model(**examples_device)
            preds, full_outputs = model(**examples_device)
            if self.args.data_type == "prsa":
                if self.scaler:
                    nexamples, seq_len, n_targets = examples_device["labels"].shape
                    preds = preds.cpu().detach().numpy()
                    preds = preds.reshape(nexamples * seq_len, n_targets)
                    preds = self.scaler.inverse_transform(preds)
                    preds = preds.reshape(
                        nexamples, 
                        seq_len, 
                        n_targets
                    )
                    labels = labels.cpu().detach().numpy()
                    labels = labels.reshape(nexamples * seq_len, n_targets)
                    labels = self.scaler.inverse_transform(labels)
                    labels = labels.reshape(
                        nexamples, 
                        seq_len, 
                        n_targets
                    )
            if self.args.data_type == "kdd":
                preds = preds.cpu().detach().numpy()
                preds = sigmoid(preds)
                labels = labels.cpu().detach().numpy()

            log.info("Inspected labels after epoch:")
            log.info(labels)
            log.info("Inspected preds after epoch:")
            log.info(preds)

 
class MultiRegressionTrainerMSE(Trainer):
    """
    Trainer class for multi-output regression tasks
    where the labels are in the range [0, 1].
    """

    def __init__(self, *args,  **kwargs):
        self.family =  kwargs.pop("family")
        self.scale_targets = kwargs.pop("scale_targets")
        self.dump_input = True
        self.script_logger =  kwargs.pop("script_logger")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Combined RMSE
        This function is also used by trainer.predict()
        """

        labels = inputs.get("labels")

        # For debugging
        if self.dump_input:
            np.savetxt(
                self.args.output_dir + '/input.txt', 
                inputs["input_ids"][0].cpu().numpy(),
                 fmt='%.0f',
            )
            np.savetxt(
                self.args.output_dir + '/labels.txt', 
                inputs["labels"][0].cpu().numpy(),
                 fmt='%.5f',
            )
            self.dump_input = False        

        preds, full_outputs = model(**inputs)
        loss_fct = torch.nn.functional.mse_loss
        loss = loss_fct(labels.flatten(), preds.flatten())

        return (loss, {"logits": preds}) if return_outputs else loss


class BinaryClassificationTrainerBCE(Trainer):
    def __init__(self, *args,  **kwargs):
            self.family =  kwargs.pop("family")
            self.dump_input = True
            self.scale_targets = kwargs.pop("scale_targets") # unused on KDD
            self.script_logger =  kwargs.pop("script_logger")
            super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Binary Cross Entropy with Logits
        This function is also used by trainer.predict()
        """

        labels = inputs.get("labels") 

        # For debugging
        if self.dump_input:
            np.savetxt(
                self.args.output_dir + '/input.txt', 
                inputs["input_ids"][0].cpu().numpy(),
                fmt='%.0f',
            )
            np.savetxt(
                self.args.output_dir + '/labels.txt', 
                np.array([inputs["labels"][0].cpu().numpy()]),
                fmt='%.5f',
            )
            self.dump_input = False        

        preds, full_outputs = model(**inputs)
        loss_fct = torch.nn.functional.binary_cross_entropy_with_logits
        imbalance = 0.1 # hard-coded
        # Class 1: 10, Class 0: 1
        weight = (imbalance * 100.) * labels.flatten()
        weight = weight + (1. * (1 - labels.flatten()))
        loss = loss_fct(preds.flatten(), labels.flatten(), weight=weight)

        return (loss, {"logits": preds}) if return_outputs else loss       


class TrainerWithInputDump(Trainer):

    def __init__(self, *args,  **kwargs):
        self.script_logger =  kwargs.pop("script_logger")
        self.dump_input = True
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Dump the inputs to a txt file once, then compute the loss as usual"""

        if self.dump_input:
            np.savetxt(
                self.args.output_dir + '/input.txt', 
                inputs["input_ids"][0].cpu().numpy(),
                 fmt='%.0f',
            )
            np.savetxt(
                self.args.output_dir + '/masked_labels.txt', 
                inputs["masked_lm_labels"][0].cpu().numpy(),
                fmt='%.0f',
            )
            self.dump_input = False

        return super().compute_loss(model, inputs, return_outputs)


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_metrics_kdd(model_output):
        preds = model_output.predictions
        preds = sigmoid(preds)
        labels = model_output.label_ids
        F1_score = f1_score(
            labels.flatten(),
            np.where(preds.flatten() < 0.5, 0, 1), 
        )
        avg_precision = average_precision_score(
            labels.flatten(),
            preds.flatten(), 
        )
        precision, recall, fscore, support = precision_recall_fscore_support(
            labels.flatten(),
            np.where(preds.flatten() < 0.5, 0, 1),
            average="binary",
        )
        score = {
            'F1': F1_score,
            'AP': avg_precision,
            'Recall': recall,
            'Precision': precision,
        }
        return score

def compute_metrics_prsa_with_scaler(scaler):
    def compute_metrics_prsa(model_output):
        preds = model_output.predictions
        labels = model_output.label_ids
        if scaler:
            nexamples, seq_len, n_targets = labels.shape
            preds = preds.reshape(nexamples * seq_len, n_targets)
            preds = scaler.inverse_transform(preds)
            labels = labels.reshape(nexamples * seq_len, n_targets)
            labels = scaler.inverse_transform(labels)
        combined_RMSE = combined_rmse(preds.flatten(), labels.flatten())
        combined_MAPE = combined_mape(preds.flatten(), labels.flatten())
        score = {
            'combined_RMSE': combined_RMSE,
            'combined_MAPE': combined_MAPE,
        }
        return score
    return compute_metrics_prsa 

def fine_tune(args, model, dataset, train_dataset, eval_dataset, test_dataset, vocab):
    # Register custom config and model
    if args.family == "fttransf_flatten":
        AutoConfig.register("FTTransformerFlatten", FTTransformerFlattenConfig)
        AutoModelForSequenceClassification.register(FTTransformerFlattenConfig, FTTransformerFlatten)
        data_collator_ft = FTTransformerFlattenDataCollatorForFineTuning(
            tokenizer=model.tokenizer, 
            ncols=dataset.ncols,
            seq_len=dataset.seq_len,
            data_type=args.data_type,
        )
    if args.family == "tabbie":
        AutoConfig.register("Tabbie", TabbieConfig)
        AutoModelForSequenceClassification.register(TabbieConfig, Tabbie)
        data_collator_ft = TabbieDataCollatorForFineTuning(
            tokenizer=model.tokenizer, 
            ncols=dataset.ncols,
            seq_len=dataset.seq_len,
            data_type=args.data_type,
        )
    elif args.family == "column_tabbert":
        AutoConfig.register("ColumnTabBERT", ColumnTabBERTConfig)
        AutoModelForSequenceClassification.register(ColumnTabBERTConfig, ColumnTabBERT)
        data_collator_ft = ColumnTabBERTDataCollatorForFineTuning(
            tokenizer=model.tokenizer, 
            ncols=dataset.ncols,
            seq_len=dataset.seq_len,
            data_type=args.data_type,
        )
    elif args.family == "row_tabbert":
        AutoConfig.register("RowTabBERT", RowTabBERTConfig)
        AutoModelForSequenceClassification.register(RowTabBERTConfig, RowTabBERT)
        data_collator_ft = RowTabBERTDataCollatorForFineTuning(
            tokenizer=model.tokenizer, 
            ncols=dataset.ncols,
            seq_len=dataset.seq_len,
            data_type=args.data_type,
        )
    elif args.family == "fieldy":
        AutoConfig.register("Fieldy", FieldyConfig)
        AutoModelForSequenceClassification.register(FieldyConfig, Fieldy)
        data_collator_ft = FieldyDataCollatorForFineTuning(
            tokenizer=model.tokenizer, 
            ncols=dataset.ncols,
            seq_len=dataset.seq_len,
            data_type=args.data_type,
        )

    # Load fine-tuning dataset if needed
    if args.data_type == "kdd":
        dataset, preserved_cols = load_kdd(args, pre_training=False)
        dataset.seq_len = 10
        vocab = dataset.vocab
        custom_special_tokens = vocab.get_special_tokens()
        train_dataset, eval_dataset, test_dataset = random_split_dataset(
            dataset,
            log,
            train_frac=0.6,
            val_frac=0.5,
            random_seed=args.seed,
        )

    # Load pre-trained model
    ft_model = AutoModelForSequenceClassification.from_pretrained(
        join(args.output_dir) + "/pt/",
        vocab=dataset.vocab,
    )
    ft_model.config.problem_type = 'multi_label_classification'

    # Target-processing 
    if args.data_type == 'prsa':
        if args.scale_targets:
            scaler, train_dataset, original_test_labels, original_train_labels = scale_targets(
                args,
                train_dataset,
                test_dataset,
                log,
            )
        # Initialize the final layer biais to the mean value of targets
        train_labels = []
        for _, labels_seq in train_dataset:
            train_labels.append(labels_seq)
        train_labels = np.stack(train_labels)
        init_preds_bias = torch.tensor(train_labels).mean().item()
        with torch.no_grad():
            if args.family == "tabbie":
                ft_model.ft_head.bias.fill_(init_preds_bias)
            else:
                ft_model.sequence_transformer.ft_head.bias.fill_(init_preds_bias)
        log.info(f"Initialized final layer bias to {init_preds_bias}")

    elif args.data_type == 'kdd':
        # No targets scaling
        scaler=None
        original_train_labels = []
        for _, labels_seq in train_dataset:
            original_train_labels.append(labels_seq)
        original_train_labels = np.stack(original_train_labels)
        original_test_labels = []
        for _, labels_seq in test_dataset:
            original_test_labels.append(labels_seq) 
        original_test_labels = np.stack(original_test_labels)
        # Initialize the final layer biais to the mean value of targets (class imbalance)
        train_labels = []
        for _, labels_seq in train_dataset:
            train_labels.append(labels_seq)
        train_labels = np.stack(train_labels)
        init_preds_bias = torch.tensor(train_labels).mean() 
        init_preds_bias = init_preds_bias.log()
        with torch.no_grad():
            if args.family == "tabbie":
                ft_model.ft_head.bias.fill_(init_preds_bias)
            else:
                ft_model.sequence_transformer.ft_head.bias.fill_(init_preds_bias)
        log.info(f"Initialized final layer bias to {init_preds_bias}")
   
    # Train
    ft_training_args = TrainingArguments(
        output_dir=args.output_dir + "/ft/",
        overwrite_output_dir=True,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=True,
        evaluation_strategy="epoch",
        prediction_loss_only=False,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.ft_epochs,
        logging_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        logging_dir=args.log_dir + "/ft/",
        label_names=["labels"],
        metric_for_best_model="eval_loss" if args.data_type == "prsa" else "eval_AP",
        greater_is_better=False if args.data_type == "prsa" else True,
        load_best_model_at_end=True,
        save_total_limit=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        learning_rate=0.00005,
    )
    if args.data_type == 'prsa':
        compute_metrics_prsa = compute_metrics_prsa_with_scaler(scaler) # Compute_metrics factory
        if args.mse:
            trainer_ft = MultiRegressionTrainerMSE(
                family=args.family,
                model=ft_model,
                args=ft_training_args,
                data_collator=data_collator_ft,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                scale_targets=args.scale_targets,
                compute_metrics=compute_metrics_prsa,
                script_logger=log,
                callbacks=[
                    InspectCallback(
                        args=args,
                        dataset=eval_dataset, 
                        logger=log, 
                        data_collator=data_collator_ft,
                        nexamples=1,
                        scaler=scaler
                    ),
                ],
            )
        else:
            pass
            log.info("Fine-tuning for PRSA needs the '--mse' flag")
    elif args.data_type == 'kdd':
        trainer_ft = BinaryClassificationTrainerBCE(
            family=args.family,
            model=ft_model,
            args=ft_training_args,
            data_collator=data_collator_ft,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            scale_targets=args.scale_targets,
            compute_metrics=compute_metrics_kdd,
            script_logger=log,
            callbacks=[
                InspectCallback(
                    args=args,
                    dataset=eval_dataset, 
                    logger=log, 
                    data_collator=data_collator_ft,
                    nexamples=1,
                    scaler=scaler
                ),
            ],
        )

    # Train
    trainer_ft.train()
    trainer_ft.save_model()

    return trainer_ft, test_dataset, original_test_labels, scaler

def format_output_dir(args, create=True):
    output_dir = args.output_dir_initial

    if args.trash:
        output_dir += "/_trash/"
    
    if args.family == "fttransf_flatten":
        output_dir += "FTTransformer"
    elif args.family == "tabbie":
        output_dir += "Tabbie"
    elif args.family == "column_tabbert":
        output_dir += "ColumnTabBert"
    elif args.family == "row_tabbert":
        output_dir += "RowTabBert"
    elif args.family == "fieldy":
        output_dir += "Fieldy"

    output_dir += f"_{args.fieldtransf_nheads}fieldtransfheads"
    output_dir += f"_{args.fieldtransf_nlayers}fieldtransflayers"
    output_dir += f"_{args.hidden_size}hs"
    output_dir += f"_{args.n_heads}heads"
    output_dir += f"_{args.n_layers}layers"

    if args.pos_emb:
        output_dir += "_posemb"
    else:
        output_dir += "_noposemb"

    if args.col_emb:
        output_dir += "_colemb"
    else:
        output_dir += "_nocolemb"

    if args.mse and args.data_type == "prsa":
        output_dir += "_MSE"
    else:
        output_dir += "_BCE"

    if args.dry_run:
        output_dir += f"_dryrun{args.dry_run}"

    if args.init_lr != 0.00005: # Non-default initial learning rate
        output_dir += f"lr{str(args.init_lr)}"

    output_dir += f"_pt{args.pt_epochs}ep_ft{args.ft_epochs}ep"

    if args.scale_targets:
        if args.scaling == "std":
            output_dir += "_stdscaledtargets"
        elif args.scaling == "minmax":
            output_dir += "_minmaxscaledtargets"
        elif args.scaling == "quantnorm":
            output_dir += "_quantnormscaledtargets"

    output_dir = output_dir + f"_seed{args.seed}"
    output_dir = output_dir + "/"
    if create:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return output_dir

def load_kdd(args, pre_training=True):
    """
    Load the KDD dataset that has already been pre-processed with ./dataset/kdd.ipynb
    """
    preserved_cols = None
    name_suffix = ""
    if pre_training:
        name_suffix = "_pt" + name_suffix
    else:
        name_suffix = "_ft" + name_suffix

    preserved_cols = None
    
    with open(f"data/kdd/KDDDataset{name_suffix}.pkl", "rb") as f:
        dataset = pickle.load(f)

    dataset.dry_run = args.dry_run

    return dataset, preserved_cols

def load_prsa(args):
    """
    Load the PRSA dataset if it has already been pre-processed.
    Otherwise, pre-process, save and load it.
    """
    
    preserved_cols = None
    name_suffix = ""
    try:
        with open(f"data/prsa/PRSADataset_labeled{name_suffix}.pkl", "rb") as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        dataset = PRSADataset(
            stride=5,
            mlm=args.mlm,
            return_labels=True,
            use_station=True,
            flatten=False,
            vocab_dir=args.vocab_dir + "/prsa/",
            timedelta=False,
            nrows=args.nrows,
            transform_date=False,
        )
        with open(f"data/prsa/PRSADataset_labeled{name_suffix}.pkl", "wb") as f:
            pickle.dump(dataset, f)

    dataset.dry_run = args.dry_run

    return dataset, preserved_cols

def prepare_labels(train_dataset, test_dataset, log):
            """
            Taking the last label in each sequence
            """
            # Save the raw test labels for final evaluation later on
            test_labels = []
            for _, labels_seq in test_dataset:
                if 1 in labels_seq:
                    label = 1
                else:
                    label = 0
                test_labels.append(label)
            test_labels = np.stack(test_labels)

            log.info(f"Label shape for one sample: {test_labels[0].shape}")

            return train_dataset, test_labels

def pre_train(args, model, dataset, train_dataset, eval_dataset, test_dataset, vocab):
    # Setup Masked Language Modeling for pre-training
        if args.family == "fttransf_flatten":
            collator_cls = "FTTransformerFlattenDataCollatorForLanguageModeling"
        elif args.family == "tabbie":
            collator_cls = "TabbieDataCollatorForLanguageModeling"
        elif args.family == "column_tabbert":
            collator_cls = "ColumnTabBERTDataCollatorForLanguageModeling"
        elif args.family == "row_tabbert":
            collator_cls = "RowTabBERTDataCollatorForLanguageModeling"
        elif args.family == "fieldy":
            collator_cls = "FieldyDataCollatorForLanguageModeling"
        log.info(f"collator class: {collator_cls}")
        
        data_collator_pt = eval(collator_cls)(
            tokenizer=model.tokenizer, 
            mlm=args.mlm, 
            mlm_probability=args.mlm_prob,
            ncols=dataset.ncols,
            seq_len=dataset.seq_len,
            data_type=args.data_type,
            randomize_seq=True,
        )

        pt_training_args = TrainingArguments(
            output_dir=args.output_dir + "/pt/",
            overwrite_output_dir=True,
            do_train=args.do_train,
            do_eval=args.do_eval,
            evaluation_strategy="epoch",
            prediction_loss_only=True,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.pt_epochs,
            logging_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            logging_dir=args.log_dir + "/pt/", 
            label_names=["masked_lm_labels"],
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            save_total_limit=1,
            label_smoothing_factor=0.0,
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            learning_rate=0.00005,
            warmup_steps=0,
        )

        trainer_pt = TrainerWithInputDump(
            script_logger=log,
            model=model.model,
            args=pt_training_args,
            data_collator=data_collator_pt,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        trainer_pt.train()
        trainer_pt.save_model()

def scale_targets(args, train_dataset, test_dataset, log):
            """
            Min-max scaling of the labels in the range [0,1] + 
            Taking the last label in each sequence
            """
            if args.scaling == "std":
                scaler = StandardScaler()
            elif args.scaling == "minmax":
                scaler = MinMaxScaler()
            elif args.scaling == "quantnorm":
                scaler = QuantileTransformer(output_distribution="normal")
            else:
                log.info(f"Missing targets scaling argument")
                sys.exit()

            # Fit the scaler on the training set
            train_labels = []
            for _, labels_seq in train_dataset:
                train_labels.append(labels_seq)
            original_train_labels = np.stack(train_labels)
            train_labels = np.concatenate(train_labels)
            scaler.fit(train_labels)

            # Save the raw test labels for final evaluation later on
            test_labels = []
            for _, labels_seq in test_dataset:
                test_labels.append(labels_seq) 
            original_test_labels = np.stack(test_labels)

            # Transform the labels using the fitted scaler
            final_targets = []
            # Targets are shared across dataset because they are subsets 
            # (i.e., views on the same shared dataset)
            all_targets = train_dataset.dataset.targets 
            for target_list in all_targets:
                transf_target_list = scaler.transform(target_list)
                final_targets.append(transf_target_list)

            # Targets are shared across dataset because they are subsets 
            # (i.e., views on the same shared dataset)
            train_dataset.dataset.targets = final_targets 

            log.info(f"Label shape for one sample: {test_labels[0].shape}")

            return scaler, train_dataset, original_test_labels, original_train_labels

def score_ft_model(args, trainer, dataset, scaler, test_labels):
    output = trainer.predict(dataset) # This outputs a numpy array

    if args.data_type == "prsa":
        preds = torch.tensor(output.predictions)
        preds = preds.numpy(force=True)
        # Scaler was fitted on two separate features
        nsamples, seq_len, noutputs = test_labels.shape
        preds = preds.reshape(nsamples * seq_len, noutputs)
        if scaler:
            preds = scaler.inverse_transform(preds)
        score_RMSE = combined_rmse(preds.flatten(), test_labels.flatten())
        score_MAPE = combined_mape(preds.flatten(), test_labels.flatten())
        score = (score_RMSE, score_MAPE)

    elif args.data_type == 'kdd':
        preds = output.predictions
        preds = sigmoid(preds)
        F1_score = f1_score(
            test_labels.flatten(),
            np.where(preds.flatten() < 0.5, 0, 1), 
        )
        avg_precision = average_precision_score(
            test_labels.flatten(),
            preds.flatten(), 
        )
        score = (F1_score, avg_precision)

    return score

def main(args):

    seed_everything(args.seed)

    # Load the selected dataset for pre-training
    if args.data_type == 'prsa':
        dataset, preserved_cols = load_prsa(args)
        num_ft_labels = dataset.seq_len * 2 # Two pollution preds for each timestep
    elif args.data_type == 'kdd':
        dataset, preserved_cols = load_kdd(args,  pre_training=True)
        num_ft_labels = 1 # Binary loan default pred for each sample
        dataset.seq_len = 10
    else:
        raise Exception(f"data type '{args.data_type}' not defined")
    vocab = dataset.vocab
    timedelta_colid = None
    custom_special_tokens = vocab.get_special_tokens()

    # Data splitting
    if args.data_type == "kdd":
        train_frac = 0.8
        val_frac = 1.0 # No testing set during pre-training
    else:
        train_frac = 0.6
        val_frac = 0.5

    train_dataset, eval_dataset, test_dataset = random_split_dataset(
        dataset,
        log,
        train_frac=train_frac,
        val_frac=val_frac,
        dry_run=args.dry_run,
        random_seed=args.seed,
    )

    # Instantiate the BERT-like model
    model = Model(
        special_tokens=custom_special_tokens,
        vocab=vocab,
        family=args.family,
        ncols=dataset.ncols,
        hidden_size=args.hidden_size,
        seq_len=dataset.seq_len,
        pos_emb=args.pos_emb,
        col_emb=args.col_emb,
        max_position_embeddings=512,
        mlm_loss=args.mlm_loss,
        n_heads=args.n_heads,
        fieldtransf_nheads=args.fieldtransf_nheads,
        fieldtransf_nlayers=args.fieldtransf_nlayers,
        n_layers=args.n_layers,
        num_ft_labels=num_ft_labels,
        dropout=args.dropout,
    )
    log.info(f"model: {model.model.__class__}")
    log.info(f"model params: {count_parameters(model.model):,}")
    log.info(f"field transf. params: {count_parameters_field_transf(model.model):,}")
    log.info(f"sequence transf. params: {count_parameters_seq_transf(model.model):,}")

    ###
    # Pre-training
    ###
    if args.pre_train:
        pre_train(
            args, 
            model, 
            dataset, 
            train_dataset, 
            eval_dataset, 
            test_dataset, 
            vocab
        )

    ###
    # Fine-tuning
    ###
    if args.fine_tune:
        trainer_ft, test_dataset, original_test_labels, scaler = fine_tune(
            args, 
            model, 
            dataset, 
            train_dataset, 
            eval_dataset, 
            test_dataset, 
            vocab
        )

        # Score
        score = score_ft_model(
            args, 
            trainer_ft, 
            test_dataset, 
            scaler=scaler, 
            test_labels=original_test_labels,
        )
        if args.data_type == "prsa":
            log.info(f"Test score (RMSE, MAPE) for {args.data_type}: {score}")
            with open(args.output_dir + "/result.txt", "w") as f:
                print(f"Test score (RMSE, MAPE) for {args.data_type}: {score}", file=f)
        elif args.data_type == "kdd":
            log.info(f"Test score (F1, AP) for {args.data_type}: {score}")
            with open(args.output_dir + "/result.txt", "w") as f:
                print(f"Test score (F1, AP) for {args.data_type}: {score}", file=f)

        # Save scaler
        if args.scale_targets:
            save_location = args.output_dir + "/ft/scaler.pkl"
            with open(save_location, "wb") as f:
                pickle.dump(scaler, f)


if __name__ == "__main__":
    
    parser = define_main_parser()
    opts = parser.parse_args()
    opts.output_dir_initial = opts.output_dir

    for run in range(opts.runs):
        opts.seed += run

        # Format output dir
        opts.output_dir = format_output_dir(opts)
        logger = logging.getLogger(__name__)
        log = logger
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            filename=opts.output_dir + f"run{run}log.txt",
            filemode="w",
        )

        # Save args
        opts.log_dir = join(opts.output_dir, "logs")
        makedirs(opts.output_dir, exist_ok=True)
        makedirs(opts.log_dir, exist_ok=True)
        with open(opts.output_dir + '/commandline_args.txt', 'w') as f:
            json.dump(opts.__dict__, f, indent=2)

        main(opts)
