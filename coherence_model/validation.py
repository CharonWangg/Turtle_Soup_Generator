import os
import pandas as pd
from utils.trainer import Trainer, Validator
import warnings
from torch.multiprocessing import Pool, Process, set_start_method, set_sharing_strategy
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # set_start_method('spawn')
    # set_sharing_strategy('file_system')
    data = pd.read_csv("./data/cmu_scifi/train_flatten.csv", index_col=0)[:100]
    # pivot start event
    start_event = data["event"].iloc[42]

    # generation
    print("0th event:", start_event)
    validator = Validator(data["event"].tolist())
    validator.super_validate(start_event)

