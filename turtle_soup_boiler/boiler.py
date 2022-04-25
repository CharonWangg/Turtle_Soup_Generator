from .utils import *
from .quantifier import Quantifier
# from utils import *
# from quantifier import Quantifier
import pickle
import string
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np

sentiment_list = ["happy", "angry", "relieving", "worrying",
                  "surprising", "anticipated", "reassuring",
                  "stressing", "calm", "sad"]


class TurtleSoupBoiler:
    # class variables
    # single_sent_prompt = 'Generate one sentence completion after given story:   '
    single_sent_prompt = "Generate a suspense story: "
    sentiment_list = ["happy", "angry", "relieving", "worrying",
                      "surprising", "anticipated", "reassuring",
                      "stressing", "calm", "sad"]
    continuation_list = ["Then,", "After a while,", "Meanwhile,", "And then,", "Some time later,", "After that,",
                         "As a result,", "Thus,"]

    def __init__(self, key=None, num_sent=5, p_sample=0.6, sample_step=1, filename=None,
                 quantifier_name="cardiffnlp/twitter-roberta-base-sentiment",
                 verbose=False):
        if key is None:
            configure_openai()
        else:
            openai.api_key = key
        self.__dict__.update(locals())
        print(self.__dict__)
        self.story = []
        self.quantifier = Quantifier(quantifier_name)
        if filename is None:
            self.filename = 'story.pkl'
        self.sent_similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.used_continuation = []

    # Generate a story by input a sentence and store it in self.story
    def generate_by_input(self):
        print('>Please input the first sentence of your story:')
        sentence = input()
        story = self.generate_story(sentence)
        self.story.append(story)
        print('>Generate story successfully!')

    # Generate a story by string and store it in self.story
    def generate_by_text(self, text):
        story = self.generate_story(text)
        self.story.append(story)
        print('>Generate story successfully!')

    # Save the story to a csv file
    def save_story(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.story, f)
            print('>Save story successfully!')

    # get the reversal of the given sentiment
    def get_reversal(self, sentiment):
        if sentiment in self.sentiment_list:
            sentiment = self.sentiment_list[-(sentiment_list.index(sentiment) + 1)]
        else:
            sentiment = "surprising"
        return sentiment

    # build the strong reversal sentence
    def get_aug_reversal(self, sent, sentiment):
        strength = self.quantifier.get_sentiment_quantity(sent)  # {sentiment: strength}
        sentiment = self.get_reversal(sentiment)
        if sentiment == "surprising":
            sentiment = sentiment.lstrip()
        else:
            sentiment = f"{strength} {sentiment}".lstrip()

        if self.verbose:
            print(sent)
            print(sentiment)

        return sentiment

    # sample the reversal probability, if it is less than p_sample, then reverse the sentence
    def get_reversal_prompt(self, curr_last_sent, curr_seq, used_continuation=[]):
        sentiment = get_sentiment(curr_last_sent)
        if self.verbose:
            print(f">Sentiment: {sentiment}")
        next_sentiment = self.get_reversal(sentiment)
        if self.verbose:
            print(f">Next Sentiment: {next_sentiment}")
        input_seq = f"{curr_seq} Then, something {next_sentiment} happened."
        if self.verbose:
            print(f">Input Sequence: {input_seq}")

        return input_seq

    def get_continuation_prompt(self, curr_seq, continuation_prompt=None):

        if continuation_prompt is None:
            # if all the continuations have been used, reset
            if len(self.used_continuation) == len(self.continuation_list):
                self.used_continuation = []
            continuation_prompt = np.random.choice(self.continuation_list)
            while continuation_prompt in self.used_continuation:
                continuation_prompt = np.random.choice(self.continuation_list)

        continuation_prompt = continuation_prompt + " something happened. "
        return f"{curr_seq} Then,"  # {continuation_prompt}"

    def generate_story(self, first_sent):
        '''
            Given the first sentence, generate a series of sentences to complete the story
        '''
        if first_sent == '':
            print('>[ERROR] Empty input sequence! Stop!')
            return ''

        first_sent = self.clean_sent(first_sent)  # .rstrip()
        curr_story = first_sent
        prev_sent = first_sent
        self.sent_lst = [first_sent]
        self.sent_emb = [self.sent_similarity_model.encode(first_sent, convert_to_tensor=True)]
        # print(self.num_sent)
        for i in range(1, self.num_sent + 1):
            if self.verbose:
                print('> Current step:', i + 1)

            # check if it is the sample step; if so, re-engineer the prompt
            # also, sample the reversal probability, if it is less than p_sample, then reverse the sentence
            if i % self.sample_step == 0 and random.random() < self.p_sample:
                gpt_input = self.get_reversal_prompt(prev_sent, curr_story)
            else:
                gpt_input = curr_story  # self.get_continuation_prompt(curr_story)
            if self.verbose:
                print(f">Input to GPT: {gpt_input}")
            # TODO: add support for multiple sub-sentences similarity check
            new_sent = self.sample_3_sent(self.gpt_get_next(gpt_input))
            # check if the new_sent is more than 3 sentences
            if not isinstance(new_sent, list):
                max_sim_scores, new_embedding = self.max_similarity(new_sent)
                if max_sim_scores > 0.9:
                    new_sent, new_embedding, max_sim_scores = self.handle_regeneration(curr_story)
                # update current sentences
                if self.verbose:
                    print('[Final new_sent]', new_sent)
                    print('[Final max_sim_scores]', max_sim_scores)
                self.sent_lst.append(new_sent)
                self.sent_emb.append(new_embedding)
                prev_sent = new_sent
            else:
                new_sent = " ".join(new_sent)
                # update current sentences
                if self.verbose:
                    print('[Final new_sent]', new_sent)
                self.sent_lst.append(new_sent)
                prev_sent = new_sent


            if self.verbose:
                print(f">Current Last Sentence: {prev_sent}")
            curr_story += f" {new_sent}"
            print()
        print('>Final story:', curr_story)
        return curr_story

    def handle_regeneration(self, curr_story):
        '''
            Handle special situation where re-generation is needed
        '''
        if self.verbose:
            print('>Need to re-generate to replace overly similar sentence!')
        re_generation_counter = 0
        max_sim_scores = 1
        while max_sim_scores > 0.9 and re_generation_counter < 10:
            gpt_input = self.get_continuation_prompt(curr_story)
            new_sent = self.gpt_get_next(gpt_input)
            max_sim_scores, new_embedding = self.max_similarity(new_sent)
            re_generation_counter += 1
            if self.verbose:
                print('[new_sent]', new_sent)
                print('[max_sim_scores]', max_sim_scores)
        if max_sim_scores > 0.9:  # if 10 prompt still cannot bring the similarity down
            print('>All the continuous prompt is not able to bring the sim score down!')
            gpt_input_lst = [self.get_continuation_prompt(curr_story, cp) for cp in self.continuation_list]
            new_sent_lst = [self.gpt_get_next(gpt_input) for gpt_input in gpt_input_lst]
            sim_score_lst = [self.max_similarity(new_sent) for new_sent in new_sent_lst]
            best_i = np.argmin([i[0] for i in sim_score_lst])
            new_sent = new_sent_lst[best_i]
            new_embedding = sim_score_lst[best_i][1]
            max_sim_scores = sim_score_lst[best_i][0]
            self.used_continuation = []
        return new_sent, new_embedding, max_sim_scores

    def gpt_get_next(self, gpt_input):
        '''
            Get the next sentence from GPT.
            Specifically designed for next-sentence generation for stories
        '''
        # get the next sentence
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.single_sent_prompt + gpt_input + '\n',
            temperature=0.7,
            max_tokens=256,
            top_p=0.7,
            frequency_penalty=0,
            presence_penalty=0,
            # stop = ['. ', '? ', '! ']
        )
        new_sent = response["choices"][0]["text"].strip("\n")
        new_sent = self.clean_sent(new_sent)
        return new_sent

    def max_similarity(self, new_sent):
        '''
            Find the max similarity with current sentences embeddings; also return the embedded new sentence
        '''
        new_embedding = self.sent_similarity_model.encode(new_sent, convert_to_tensor=True)
        sim_scores = [float(util.pytorch_cos_sim(new_embedding, sent_emb)) for sent_emb in self.sent_emb]
        return max(sim_scores), new_embedding

    def sample_3_sent(self, sents):
        '''
            Sample 3 sentences from the story
        '''
        # sample 3 sentences
        sent_list = sent_tokenize(sents)
        if len(sent_list) == 1:
            return sent_list[0]
        else:
            return sent_list[:3]

    def clean_sent(self, sent):
        '''
            Clean a given sentence
        '''
        # basic cleanup 
        sent = ''.join([c for c in sent if c in string.printable])
        sent = re.sub(' +', ' ', sent)
        sent = re.sub('\n+', ' ', sent)
        sent = re.sub('\t+', ' ', sent)
        sent = sent.strip()

        # add period, if needed
        if sent[-1] not in [".", ",", "?", "!", "'", '"']:
            sent += '.'
        return sent.capitalize()
