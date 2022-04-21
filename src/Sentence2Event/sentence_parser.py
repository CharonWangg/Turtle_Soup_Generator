# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import pickle

import pandas as pd

from eventmaker_singleSent import eventMaker
from memoryGraph_scifi import MemoryGraph

# import nltk
# nltk.download('omw-1.4')

pick = open("data/names-percentage.pkl", 'rb')  # US census data
gender_list = pickle.load(pick)  # gender_list[name] = set(genders)
pick.close()
del (pick)

memory = MemoryGraph(gender_list)

df = pd.read_csv("/home/charon/project/Turtle_Soup/data/cmu_scifi/train.csv")  # US census data

next_event = "Thor is informed by Penegal of the Asgard High Council."  # df["answer"].iloc[0]

for i in range(len(df)):
    print("*" * 200)
    next_event = df["sent"].iloc[i]
    if next_event[-1] != ".":
        next_event += "."
    try:
        eventifier = eventMaker(next_event, gender_list, memory)
        event, gen_event, ns = eventifier.getEvent()
        # print("Sentence: " + next_event)
        print("Event: " + " ".join(event))
        print("Generalized Event: " + " ".join(gen_event))
    except:
        print("Sorry, I can't find an event in that sentence. Can you try another sentence?")
        continue

    memory.add_event(event, gen_event, ns)
    print("Named Entity Dictionary: " + str(memory.NEnums))
