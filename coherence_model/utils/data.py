import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import ast
import numpy as np
import torch
import torch.utils as utils
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml

Config = yaml.safe_load(open("./config/config.yaml"))
tokenizer = AutoTokenizer.from_pretrained(Config["TOKENIZE"]["TOKENIZER_NAME"])


def load_cmu_scifi():
    dataset = load_dataset("lara-martin/Scifi_TV_Shows")
    return dataset


def str2list(string):
    l = ast.literal_eval(string)
    return l


def parse_event(tokenizer, event):
    args = str2list(event)
    args = [tokenize(tokenizer, arg) for arg in args]

    info = {}
    info["input_ids"] = [arg["input_ids"] for arg in args]
    info["attention_mask"] = [arg["attention_mask"] for arg in args]
    return info


def numpy_split_df(df, split_num):
    lst_index = list(map(lambda a: a.tolist(), np.array_split(df.index.tolist(), split_num)))
    chunk_list = []
    for idx in lst_index:
        df_split = df.loc[idx[0]: idx[-1] + 1]
        chunk_list.append(df_split)
    return chunk_list


def apply_parallel(df, func, num_cpu, valid=False):
    # divide events into chunks
    chunk_list = numpy_split_df(df, num_cpu * 100)
    examples_list = Parallel(n_jobs=num_cpu)(delayed(func)(split_df, df, valid=valid) for split_df in
                                             tqdm(chunk_list, desc="Parallel", total=len(chunk_list)))
    return examples_list


def super_make_examples(df, valid=False):
    num_cpu = cpu_count()
    examples_list = apply_parallel(df, make_examples, num_cpu, valid)
    examples = sum(examples_list, [])
    return examples


def df_split(df):
    '''
    x: time series array ()
    :return:
    '''
    train, valid = train_test_split(df, train_size=Config["DATA"]["TRAIN_SIZE"], shuffle=False)
    return train, valid


# TODO Sample a following event from same story
def hard_positive_sample(df, idx):
    story_num = df.loc[idx]["story_num"]
    # exclude input event and other story
    df = df[df["story_num"] == story_num]
    try:
        sample = df.loc[idx + 1]["event"]
    except:
        sample = df.loc[idx]["event"]
    return sample


# TODO Randomly sample a event from same story
def soft_positive_sample(df, idx):
    story_num = df.loc[idx]["story_num"]
    # exclude input event and other story
    df = df.drop(index=idx)
    df = df[df["story_num"] == story_num]
    sample = df.sample(n=1)["event"].values[0]
    return sample


# TODO Randomly sample a event from other story
def hard_negative_sample(df, idx):
    story_num = df.loc[idx]["story_num"]
    # exclude the same story
    df = df[df["story_num"] != story_num]
    sample = df.sample(n=1)["event"].values[0]
    return sample


# TODO Randomly sample a event from same story
def soft_negative_sample(df, idx):
    story_num = df.loc[idx]["story_num"]
    # exclude input event and other story
    df = df.drop(index=[idx, idx + 1])
    df = df[df["story_num"] == story_num]
    sample = df.sample(n=1)["event"].values[0]
    return sample


def tokenize(tokenizer, sentence, config=Config):
    tokens_info = tokenizer(sentence,
                            padding='max_length',
                            truncation=True,
                            max_length=config["TOKENIZE"]["MAX_SEQ_LEN"],
                            )
    return tokens_info


def make_examples(group, df=None, valid=False, config=Config):
    examples = []

    df["event"] = df["event"].str.lstrip()

    # for training --》 No Negative Sample
    # if not valid:
    for i in range(group["index"].min(), group["index"].max() + 1):
        ## Only use Duplicate Samples
        example = {"anchor": {}, "positive": {}, "negative": {}}
        # anchor
        anchor = str2list(df["event"].loc[i])
        tokens_info = [tokenize(tokenizer, anchor[j]) for j in range(len(anchor))]
        example["anchor"]["input_ids"] = [tokens_info[j]["input_ids"] for j in range(len(tokens_info))]
        example["anchor"]["attention_mask"] = [tokens_info[j]["attention_mask"] for j in range(len(tokens_info))]
        # positive
        positive = str2list(hard_positive_sample(df, i))
        tokens_info = [tokenize(tokenizer, positive[j]) for j in range(len(positive))]
        example["positive"]["input_ids"] = [tokens_info[j]["input_ids"] for j in range(len(tokens_info))]
        example["positive"]["attention_mask"] = [tokens_info[j]["attention_mask"] for j in range(len(tokens_info))]
        # hard_negative
        negative = str2list(hard_negative_sample(df, i))
        tokens_info = [tokenize(tokenizer, negative[j]) for j in range(len(negative))]
        example["negative"]["input_ids"] = [tokens_info[j]["input_ids"] for j in range(len(tokens_info))]
        example["negative"]["attention_mask"] = [tokens_info[j]["attention_mask"] for j in range(len(tokens_info))]

        # label
        # example["label"] = df["coherence"].iloc[i]
        examples.append(example)
    # else:
    #     # for valid --》 Have Negative Sample
    #     for i in tqdm(range(df.shape[0])):
    #         example = {"query": {}, "aug_query": {}}
    #         # input
    #         # more toxic
    #         tokens_info = tokenize(tokenizer, df["query"].iloc[i])
    #         example["query"]["input_ids"] = tokens_info["input_ids"]
    #         example["query"]["attention_mask"] = tokens_info["attention_mask"]
    #         # less toxic
    #         tokens_info = tokenize(tokenizer, df["aug_query"].iloc[i])
    #         example["aug_query"]["input_ids"] = tokens_info["input_ids"]
    #         example["aug_query"]["attention_mask"] = tokens_info["attention_mask"]

    # label
    # example["label"] = df["label"].iloc[i]
    # examples.append(example)

    return examples


class Event_Dataset(utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        anchor = [{
            'input_ids': torch.tensor(pair["anchor"]["input_ids"][i], dtype=torch.long),
            'attention_mask': torch.tensor(pair["anchor"]["attention_mask"][i], dtype=torch.long)
        } for i in range(len(pair["anchor"]["input_ids"]))]
        positive = [{
            'input_ids': torch.tensor(pair["positive"]["input_ids"][i], dtype=torch.long),
            'attention_mask': torch.tensor(pair["positive"]["attention_mask"][i], dtype=torch.long)
        } for i in range(len(pair["positive"]["input_ids"]))]
        negative = [{
            'input_ids': torch.tensor(pair["negative"]["input_ids"][i], dtype=torch.long),
            'attention_mask': torch.tensor(pair["negative"]["attention_mask"][i], dtype=torch.long)
        } for i in range(len(pair["negative"]["input_ids"]))]

        item = {
                "anchor": anchor,
                "positive": positive,
                "negative": negative
                }

        return item, 0

    def __len__(self):
        return len(self.pairs)


# Only query event and searching event
class Search_Dataset(utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        anchor = [{
            'input_ids': torch.tensor(pair["anchor"]["input_ids"][i], dtype=torch.long),
            'attention_mask': torch.tensor(pair["anchor"]["attention_mask"][i], dtype=torch.long)
        } for i in range(len(pair["anchor"]["input_ids"]))]

        positive = [{
            'input_ids': torch.tensor(pair["positive"]["input_ids"][i], dtype=torch.long),
            'attention_mask': torch.tensor(pair["positive"]["attention_mask"][i], dtype=torch.long)
        } for i in range(len(pair["positive"]["input_ids"]))]

        item = {"anchor": anchor, "positive": positive}
        return item

    def __len__(self):
        return len(self.pairs)


def make_event_loader(df, valid=False):
    if not valid:
        examples = super_make_examples(df, valid=valid)
        t_dataset = Event_Dataset(examples)
        loader = utils.data.DataLoader(t_dataset, batch_size=Config["DATA"]["TRAIN_BATCH_SIZE"],
                                       shuffle=True, pin_memory=True, num_workers=Config["DATA"]["NUM_WORKERS"])
    else:
        examples = super_make_examples(df, valid=valid)
        t_dataset = Event_Dataset(examples)
        loader = utils.data.DataLoader(t_dataset, batch_size=Config["DATA"]["VALID_BATCH_SIZE"],
                                       shuffle=True, pin_memory=True, num_workers=Config["DATA"]["NUM_WORKERS"])

    return loader


# repeat the start event to align with event list
def make_search_loader(event, event_list):
    tokenizer = AutoTokenizer.from_pretrained(Config["TOKENIZE"]["TOKENIZER_NAME"])
    example = parse_event(tokenizer, event)
    event_pairs = [{"anchor": example, "positive": parse_event(tokenizer, event)} for event in tqdm(event_list, desc="Tokenizing")]
    # print("tokenizing done")
    search_loader = utils.data.DataLoader(Search_Dataset(event_pairs), batch_size=Config["DATA"]["VALID_BATCH_SIZE"],
                                          shuffle=False, pin_memory=True, num_workers=Config["DATA"]["NUM_WORKERS"])
    return search_loader
