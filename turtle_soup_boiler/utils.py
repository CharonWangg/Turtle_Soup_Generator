import os
import random

import openai


def check_path(path):
    if os.path.exists(path):
        return True
    else:
        return False


def configure_openai():
    print('>Please input your OpenAI API key:')
    # openai.api_key = input()
    openai.api_key = "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"


# get the sentiment of the given sentence
def get_sentiment(sequence):
    input_seq = f"'{sequence}' What is the sentiment of the sentence, happy, sad, angry, calm, relieving, stressing, " \
                f"worrying, reassuring, surprising, or anticipated? "
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=input_seq,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["."]
    )

    res = response["choices"][0]["text"].split(" ")[-1].strip(".").strip("\n")
    return res


# get the reversal of the given sentiment
def get_reversal(sentiment):
    sentiment_list = ["happy", "angry", "relieving", "worrying", "surprising", "anticipated", "reassuring", "stressing",
                      "calm", "sad"]
    if sentiment in sentiment_list:
        return sentiment_list[-(sentiment_list.index(sentiment) + 1)]
    else:
        return "surprising"


# sample the reversal probability, if it is less than p_sample, then reverse the sentence
def reversal_sample(curr_last_sent, curr_seq, p_sample=0.6, verbose=False):
    if random.random() < p_sample:
        sentiment = get_sentiment(curr_last_sent)
        if verbose:
            print(f">Sentiment: {sentiment}")
        next_sentiment = get_reversal(sentiment)
        if verbose:
            print(f">Next Sentiment: {next_sentiment}")
        input_seq = f"{curr_seq} Then, something {next_sentiment} happened."
        if verbose:
            print(f">Input Sequence: {input_seq}")
    else:
        # follow the original sequence
        input_seq = f"{curr_seq} Then,"
        if verbose:
            print(f">Input Sequence: {input_seq}")

    return input_seq


# # generate story from the given sequence
# def generate_story(input, num_sent=5, p_sample=0.6, sample_step=2, verbose=False):
#     curr_seq = input
#     curr_last_sent = input
#     for i in range(1, num_sent + 1):
#         # check if it is the sample step
#         if i % sample_step == 0:
#             # sample the reversal probability, if it is less than p_sample, then reverse the sentence
#             input_seq = reversal_sample(curr_last_sent, curr_seq, p_sample, verbose)
#         else:
#             input_seq = f"{curr_seq} Then,"
#             if verbose:
#                 print(f">Input Sequence: {input_seq}")
#         # get the next sentence
#         response = openai.Completion.create(
#             model="text-davinci-002",
#             prompt=input_seq,
#             temperature=0.7,
#             max_tokens=256,
#             top_p=0.7,
#             frequency_penalty=0,
#             presence_penalty=0.2,
#             # stop=[".", "?", "!"]
#         )
#         res = response["choices"][0]["text"].strip("\n")
#         curr_last_sent = res
#         if verbose:
#             print(f">Current Last Sentence: {curr_last_sent}")
#         curr_seq += f" {res}"
#         if verbose:
#             print(f">Current Sequence: {curr_seq}")

#     return curr_seq
