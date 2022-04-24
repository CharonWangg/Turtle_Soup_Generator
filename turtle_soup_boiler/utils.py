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
    openai.api_key = input()


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


# generate story from the given sequence
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
#             presence_penalty=0
#         )
#         res = response["choices"][0]["text"].strip("\n")
#         curr_last_sent = res
#         if verbose:
#             print(f">Current Last Sentence: {curr_last_sent}")
#         curr_seq += f" {res}"
#         if verbose:
#             print(f">Current Sequence: {curr_seq}")
#
#     return curr_seq


def get_continuation(used_list=[]):
    continuation_list = ["Then,", "After a while,", "Meanwhile,", "And then,", "Some time later,", "After that,",
                         "As a result,", "Thus,"]
    final_choices = [cont for cont in continuation_list if cont not in used_list]
    return "" #"Then," #random.choice(final_choices)
