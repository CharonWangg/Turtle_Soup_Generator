from random import sample
from utils import *
import pickle
import string
import re 


class TurtleSoupBoiler:
    # class variables
    single_sent_prompt = 'Generate one sentence completion for the following sentence: \n'


    def __init__(self, num_sent=5, p_sample=0.6, sample_step=2, filename=None, verbose=False):
        configure_openai()
        self.__dict__.update(locals())
        self.story = []
        if filename is None:
            self.filename = 'story.pkl'
        else:
            self.filename = filename
        self.num_sent = num_sent
        self.p_sample = p_sample
        self.sample_step = sample_step
        self.verbose = verbose

    # Generate a story by input a sentence and store it in self.story
    def generate_by_input(self):
        if self.verbose:
            print('>Please input the first sentence of your story:')
        
        sentence = input()
        # sentence = 'Jack went home with a new cat.'
        story = self.generate_story(sentence)
        self.story.append(story)
        if self.verbose:
            print('>Generate story successfully!')
            print(self.story)

    # # Generate a story by string and store it in self.story
    # def generate_by_text(self, text, verbose=False):
    #     story = generate_story(text, self.num_sent, self.p_sample, self.sample_step, verbose)
    #     self.story.append(story)
    #     print('>Generate story successfully!')

    # Save the story to a csv file
    def save_story(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.story, f)
            print('>Save story successfully!')

    def generate_story(self, first_sent):
        '''
            Given the first sentence, generate a series of sentences to complete the story
        '''
        if first_sent == '':
            print('>[ERROR] Empty input sequence! Stop!')
            return ''

        curr_seq = first_sent
        curr_last_sent = first_sent
        for i in range(1, self.num_sent + 1):
            if self.verbose:
                print('> Current step:', i + 1)
            
            # check if it is the sample step; if so, re-engineer the prompt
            if i % self.sample_step == 0:
                # sample the reversal probability, if it is less than p_sample, then reverse the sentence
                input_seq = reversal_sample(curr_last_sent, curr_seq, self.p_sample, verbose)
            else:
                input_seq = f"{curr_seq} Then,"
                if self.verbose:
                    print(f">Input Sequence: {input_seq}")
            
            # get the next sentence
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=self.single_sent_prompt + input_seq,
                temperature=0.7,
                max_tokens=256,
                top_p=0.7,
                frequency_penalty=0,
                presence_penalty=0.2
            )
            res = response["choices"][0]["text"].strip("\n")
            res = self.clean_sent(res)
            curr_last_sent = res
            if self.verbose:
                print(f">Current Last Sentence: {curr_last_sent}")
            curr_seq += f" {res}"
            if self.verbose:
                print(f">Current Sequence: {curr_seq}")

        return curr_seq


    def gpt_helper(self, prompt):
        pass

    def clean_sent(self, sent):
        '''
            Clean a given sentence
        '''
        sent = ''.join([c for c in sent if c in string.printable])
        sent = sent.strip()
        sent = re.sub(' +', ' ', sent)
        sent = re.sub('\n+', '\n', sent)
        sent = re.sub('\t+', ' ', sent)
        return sent