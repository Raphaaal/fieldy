import random
import numpy as np
import torch

class ddict(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def combined_rmse(y, y_pred):
    """
    Combined RMSE when outputting two targets (PRSA dataset)
    """

    # Reshape after the flatten
    y = y.reshape(-1, 2) 
    y_pred = y_pred.reshape(-1, 2) 
    
    y_1 = y[:, 0]
    y_2 = y[:, 1]

    y_pred_1 = y_pred[:, 0]
    y_pred_2 = y_pred[:, 1]

    s_error_1 = np.power(y_1 - y_pred_1, 2)
    s_error_2 = np.power(y_2 - y_pred_2, 2)

    combined_avg = (s_error_1 + s_error_2) / 2.
    combined_rmse = np.sqrt(np.mean(combined_avg))

    return combined_rmse

def combined_rmse_torch(y, y_pred):
    """
    Combined RMSE when outputting two targets (PRSA dataset)
    """
    
    # Reshape after the flatten
    y = y.reshape(-1, 2) 
    y_pred = y_pred.reshape(-1, 2) 
    
    y_1 = y[:, 0]
    y_2 = y[:, 1]

    y_pred_1 = y_pred[:, 0]
    y_pred_2 = y_pred[:, 1]

    s_error_1 = torch.pow(y_1 - y_pred_1, 2)
    s_error_2 = torch.pow(y_2 - y_pred_2, 2)

    combined_avg = (s_error_1 + s_error_2) / 2.
    combined_rmse = torch.sqrt(torch.mean(combined_avg))

    return combined_rmse

def combined_mape(y, y_pred):
    """
    Combined MAPE when outputting two targets (PRSA dataset)
    """

    # Reshape after the flatten
    y = y.reshape(-1, 2) 
    y_pred = y_pred.reshape(-1, 2) 
    
    y_1 = y[:, 0]
    y_2 = y[:, 1]

    y_pred_1 = y_pred[:, 0]
    y_pred_2 = y_pred[:, 1]

    s_error_1 = np.abs(y_1 - y_pred_1) / np.abs(y_1)
    s_error_2 = np.abs(y_2 - y_pred_2) / np.abs(y_2)

    combined_mape = (s_error_1 * 100.0 + s_error_2 * 100.0) / 2.0

    return np.mean(combined_mape)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_field_transf(model):
    return sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and "field" in n))

def count_parameters_seq_transf(model):
    return sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and "seq" in n))

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def random_split_dataset(dataset, log, train_frac=0.6, val_frac=0.5, random_seed=20200706, dry_run=0):

    log.info(f"Input # columns: {dataset[0][0].shape[1]}")
    log.info(f"Input max sequence length: {dataset[0][0].shape[0]}")

    # Set the train - val - test fractions
    totalN = len(dataset)
    if dry_run == 3:
        trainN = 1
        valN = 1
        testN = 1
    else:
        trainN = int(train_frac * totalN)
        valtestN = totalN - trainN
        valN = int(valtestN * val_frac)
        testN = valtestN - valN
        assert totalN == trainN + valN + testN
    lengths = [trainN, valN, testN]
    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("% lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(
        trainN / totalN, 
        valN / totalN,
        testN / totalN)
    )

    # state snapshot
    state = {}
    state['seeds'] = {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(),
        'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

    # seed
    seed_everything(random_seed)

    # Split
    (
        train_dataset, 
        eval_dataset, 
        test_dataset
    ) = torch.utils.data.dataset.random_split(dataset, lengths)

    # reinstate state
    random.setstate(state['seeds']['python_state'])
    np.random.set_state(state['seeds']['numpy_state'])
    torch.set_rng_state(state['seeds']['torch_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['seeds']['cuda_state'])

    return train_dataset, eval_dataset, test_dataset

def seed_everything(seed: int):
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda