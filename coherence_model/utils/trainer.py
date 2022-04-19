import sys
import os
sys.path.append("./")

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import yaml
from utils.metric import Metric
from utils.model import MODEL
from utils.data import make_search_loader

config = yaml.safe_load(open("./config/config.yaml"))


# Loss
def loss_func(pp_pair, pn_pair, m=1.0):
    basic_loss = pp_pair - pn_pair + m
    loss = torch.max(torch.tensor(0).to(config["DEVICE"]), basic_loss)

    return loss


# AUROC
def roc(pred, label):
    roc = roc_auc_score(label, pred, multi_class="ovr")
    return roc


# TODO
# Scheduler
# def scheduler(optimizer, warm_steps, num_training_steps):
#     scler = transformers.get_cosine_schedule_with_warmup(optimizer,
#                                                          num_warmup_steps=warm_steps,
#                                                          num_training_steps=num_training_steps,
#                                                          )
#     return scler

# Simple Parameter Setting
def optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["OPTIMIZATION"]["LR"])
    return optimizer


def checkpoint_storage(model):
    if not os.path.exists(config["LOG"]["PATH"]["MODEL_SAVE_PATH"]):
        os.makedirs(config["LOG"]["PATH"]["MODEL_SAVE_PATH"])
    torch.save(model.state_dict(), config["LOG"]["PATH"]["MODEL_SAVE_PATH"] + "/model_best.bin")

# Trainer
class Trainer():
    def __init__(self, train_loader, valid_loader, path=None):
        self.model = checkpoint_call(path)
        print(self.model)
        self.optimizer = optimizer(self.model)

        num_training_steps = len(train_loader) / config["OPTIMIZATION"]["ACC_GRADIENT_STEPS"] * config["OPTIMIZATION"][
            "NUM_EPOCHS"]
        warm_steps = int(num_training_steps * config["OPTIMIZATION"]["WARM_FRAC"])
        print("*" * 100)
        print(f"The total traning steps is {num_training_steps}.")
        print(f"The warmup steps is {warm_steps}.")
        print("*" * 100)

        if config["CUDA"]["FP16"]:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level=config["CUDA"]["FP16_OPT_LEVEL"])
        # self.scler = scheduler(self.optimizer, warm_steps, num_training_steps)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self):
        epoch = 0
        patience = 0
        best_acc = 0
        metric = Metric(config["LOG"]["PATH"]["TRAIN_LOSS_PATH"])

        while epoch < config["OPTIMIZATION"]["NUM_EPOCHS"]:
            self.model.train()
            for step, (data, label) in enumerate(self.train_loader):

                anchor = [{key: data["anchor"][i][key].to(config["DEVICE"]) for key in data["anchor"][i].keys() if
                         key in ["input_ids", "attention_mask"]} for i in range(len(data["anchor"]))]
                positive = [{key: data["positive"][i][key].to(config["DEVICE"]) for key in data["positive"][i].keys() if
                         key in ["input_ids", "attention_mask"]} for i in range(len(data["positive"]))]
                negative = [{key: data["negative"][i][key].to(config["DEVICE"]) for key in data["negative"][i].keys() if
                         key in ["input_ids", "attention_mask"]} for i in range(len(data["negative"]))]

                loss = self.model(anchor, positive, negative)
                loss = loss / config["OPTIMIZATION"]["ACC_GRADIENT_STEPS"]

                if config["CUDA"]["FP16"]:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                metric.step(loss.item())

                if step % config["OPTIMIZATION"]["ACC_GRADIENT_STEPS"] == 0 or step == len(train_loader) - 1:
                    self.optimizer.step()
                    # self.scler.step()
                    self.optimizer.zero_grad()

                if step % config["LOG"]["STEP"]["LOG_STEPS"] == 0:
                    print("Epoch: {}\t||\tStep: {}/{}\t||\tAverage loss: {}\t||Max loss: {}\t||\t Min loss: {}".format(
                        epoch, step, len(self.train_loader), metric.ave, metric.max, metric.min))
                    metric.log("train_loss")
                    metric.reset()
                # if step != 0 and step % config["LOG"]["STEP"]["VALID_STEPS"] == 0:
                #     loss, acc = self.validate()
                #     print(
                #         "Epoch: {}\t||\tStep: {}/{}\t||\tValidation AUCROC Score: {}\t||Max loss: {}\t||\t Min loss: {}".format(
                #             epoch, step, len(self.train_loader), acc, loss, loss))
                #     if acc <= best_acc:
                #         patience += 1
                #     else:
                #         best_acc = acc
                #         checkpoint_storage(self.model)
                #         patience = 0
                #     if patience == config["PATIENCE"]:
                #         print("Early stopping has reached!")
                #         break

                del data, anchor, positive, negative, loss,
                torch.cuda.empty_cache()

            # loss, acc = self.validate()
            # print(
            #     "Epoch: {}\t||\tStep: {}/{}\t||\tValidation AUCROC Score: {}\t||Max loss: {}\t||\t Min loss: {}".format(
            #         epoch, step, len(self.train_loader), acc, loss, loss))
            # if acc <= best_acc:
            #     patience += 1
            # else:
            #     best_acc = acc
            #     checkpoint_storage(self.model)
            #     patience = 0
            # if patience == config["OPTIMIZATION"]["PATIENCE"]:
            #     print("Early stopping has reached!")
            #     break
            checkpoint_storage(self.model)    # save model
            epoch += 1

        return self.model

    # Validation
    def evaluate(self):
        losses = []
        correct = 0
        metric = Metric(config["LOG"]["PATH"]["VALID_LOSS_PATH"])
        # torch.save(model.state_dict(),"drive/MyDrive/checkpoint1.bin")
        with torch.no_grad():
            for step, (data, label) in enumerate(self.valid_loader):
                t1, t2 = data["t1"].to(config["DEVICE"]), data["t2"].to(config["DEVICE"])
                label = label.cuda()

                pred = self.model(t1, t2)
                loss = loss_func(pred, label)
                correct += roc(pred, label)
                losses.append(loss.item())

                del data, pred, label, loss
                torch.cuda.empty_cache()

        metric.log("valid_loss")
        print("Validation--------------> Loss: {} | Accuracy: {}".format(np.mean(losses),
                                                                         correct / len(self.valid_loader)))

    # TODO
    def validate(self, raw=False):
        preds = []
        ground_truth = []
        losses = []
        # metric = Metric(config["LOG"]["PATH"]["VALID_LOSS_PATH"])
        # torch.save(model.state_dict(),"drive/MyDrive/checkpoint1.bin")
        with torch.no_grad():
            for step, (data, label) in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                t1, t2 = data["t1"].to(config["DEVICE"]), data["t2"].to(config["DEVICE"])
                label = label.cuda()

                pred = self.model(t1, t2)
                loss = loss_func(pred, label)
                pred = F.softmax(pred, dim=-1)
                preds.extend(pred.cpu().numpy())
                ground_truth.extend(label.cpu().numpy())
                losses.append(loss.item())

                del data, pred, label, loss
                torch.cuda.empty_cache()

        # metric.log("valid_loss")
        if raw:
            return preds, np.mean(losses), roc(np.stack(preds), ground_truth)
        else:
            return np.mean(losses), roc(np.stack(preds), ground_truth)

class Validator():
    def __init__(self, event_list):
        self.model = checkpoint_call(config["LOG"]["PATH"]["MODEL_SAVE_PATH"]+"/model_best.bin")
        self.event_list = event_list

    # TODO : generate 5 events
    def simple_validate(self, start_event):
        sim_list = []

        for event in tqdm(self.event_list):
            # caluculate the similarity
            sim = self.model.calculate_similarity(start_event, event).detach().cpu().numpy().squeeze()
            sim_list.append(sim)

        # sort the similarity
        sorted_sim = np.argsort(np.stack(sim_list))[::-1][:5]  # get the index of the 10 most similar events
        # get the most similar events
        most_similar_events = [self.event_list[i] for i in sorted_sim]
        sim_list = [sim_list[i] for i in sorted_sim]

        print("start_event : {}".format(start_event))
        print("Most similar events: {}".format(most_similar_events))
        print("similarity: {}".format(sim_list))

        return 0

    def super_validate(self, start_event, event_num=5):
        # greedy search for event
        current_event = start_event
        self.model.eval()
        with torch.no_grad():
            for i in range(event_num):
                search_loader = make_search_loader(current_event, self.event_list)
                sim_list = self.model.super_calculate_similarity(search_loader)
                sorted_sim = np.argsort(np.stack(sim_list))[::-1]  # get the most coherent event
                if current_event == self.event_list[sorted_sim[0]]:
                    current_event = self.event_list[sorted_sim[1]]
                else:
                    current_event = self.event_list[sorted_sim[0]]

                print("{}th event: {}".format(i+1, current_event))


# make model
def checkpoint_call(path=None, cuda=True):
    if path:
        model = MODEL()

        state_dict = torch.load(path)
        for n, p in model.named_parameters():
            if n not in state_dict.keys():
                state_dict[n] = p

        drop_list = []
        for n, p in state_dict.items():
            if n not in model.state_dict().keys():
                drop_list.append(n)

        for i in drop_list:
            state_dict.pop(i)

        model.load_state_dict(state_dict)
        # model._init_weights(model.qa_outputs)
    else:
        model = MODEL()

    return model.to(config["DEVICE"]) if cuda == True else model.cpu()
