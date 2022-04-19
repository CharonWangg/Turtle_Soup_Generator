import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from utils.data import load_cmu_scifi, df_split, make_event_loader
from utils.util import fix_all_seeds
from utils.trainer import Trainer
import yaml

if __name__ == "__main__":
    config = yaml.safe_load(open("./config/config.yaml"))
    data = pd.read_csv("./data/cmu_scifi/train_flatten.csv", index_col=0)[:24000]

    fix_all_seeds(config["SEED"])  # fix all seeds for reproducibility of results
    train, valid = df_split(data)
    train_loader = make_event_loader(data, config["DATA"]["TRAIN_BATCH_SIZE"])
    valid_loader = make_event_loader(valid, config["DATA"]["VALID_BATCH_SIZE"])

    trainer = Trainer(train_loader, valid_loader)
    trainer.train()
