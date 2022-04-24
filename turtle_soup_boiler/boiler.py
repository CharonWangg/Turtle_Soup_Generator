from .utils import *
from .quantifier import Quantifier
import pickle
import string
import re
from sentence_transformers import SentenceTransformer, util

sentiment_list = ["happy", "angry", "relieving", "worrying",
                  "surprising", "anticipated", "reassuring",
                  "stressing", "calm", "sad"]


class TurtleSoupBoiler:
    # class variables
    single_sent_prompt = 'Generate one sentence completion after given story:   '
    sentiment_list = ["happy", "angry", "relieving", "worrying",
                      "surprising", "anticipated", "reassuring",
                      "stressing", "calm", "sad"]

    def __init__(self, key=None, num_sent=5, p_sample=0.6, sample_step=1, filename=None,
                 quantifier_name="cardiffnlp/twitter-roberta-base-sentiment",
                 verbose=False):
        if key is None:
            configure_openai()
        else:
            openai.api_key = key
        self.__dict__.update(locals())
        self.story = []
        self.quantifier = Quantifier(quantifier_name)
        if filename is None:
            self.filename = 'story.pkl'
        self.sent_similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
    def reversal_sample(self, curr_last_sent, curr_seq, used_continuation=[]):
        if random.random() < self.p_sample:
            sentiment = get_sentiment(curr_last_sent)
            if self.verbose:
                print(f">Sentiment: {sentiment}")
            next_sentiment = self.get_aug_reversal(curr_last_sent, sentiment)
            if self.verbose:
                print(f">Next Sentiment: {next_sentiment}")
            input_seq = f"{curr_seq} Then, something {next_sentiment} happened."
            if self.verbose:
                print(f">Input Sequence: {input_seq}")
        else:
            # follow the original sequence
            continuation = get_continuation()
            input_seq = f"{curr_seq} {continuation}"

            if self.verbose:
                print(f">Input Sequence: {input_seq}")

        return input_seq

    def generate_story(self, first_sent):
        '''
            Given the first sentence, generate a series of sentences to complete the story
        '''
        if first_sent == '':
            print('>[ERROR] Empty input sequence! Stop!')
            return ''

        first_sent = first_sent.rstrip()
        curr_seq = first_sent
        curr_last_sent = first_sent
        self.sent_lst = [first_sent]
        self.sent_emb = [self.sent_similarity_model.encode(first_sent, convert_to_tensor=True)]
        
        for i in range(1, self.num_sent + 1):
            if self.verbose:
                print('> Current step:', i + 1)

            # check if it is the sample step; if so, re-engineer the prompt
            if i % self.sample_step == 0:
                # sample the reversal probability, if it is less than p_sample, then reverse the sentence
                input_seq = self.reversal_sample(curr_last_sent, curr_seq, self.p_sample, self.verbose)
            else:
                continuation = get_continuation()
                input_seq = f"{curr_seq} {continuation}"
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
                presence_penalty=0
            )
            res = response["choices"][0]["text"].strip("\n")
            res = self.clean_sent(res)

            max_sim_scores, new_embedding = self.max_similarity(res)
            print('[res]', res)
            print('[max_sim_scores]', max_sim_scores)
            self.sent_lst.append(res)
            self.sent_emb.append(new_embedding)

            curr_last_sent = res
            if self.verbose:
                print(f">Current Last Sentence: {curr_last_sent}")
            curr_seq += f" {res}"
            if self.verbose:
                print(f">Current Sequence: {curr_seq}")

        return curr_seq

    def max_similarity(self, new_sent):
        '''
            Find the max similarity with current sentences embeddings; also return the embedded new sentence
        '''
        new_embedding = self.sent_similarity_model.encode(new_sent, convert_to_tensor=True)
        sim_scores = [float(util.pytorch_cos_sim(new_embedding, sent_emb)) for sent_emb in self.sent_emb]
        return max(sim_scores), new_embedding
        
    def clean_sent(self, sent):
        '''
            Clean a given sentence
        '''
        sent = ''.join([c for c in sent if c in string.printable])
        sent = sent.strip()
        sent = re.sub(' +', ' ', sent)
        sent = re.sub('\n+', '\n', sent)
        sent = re.sub('\t+', ' ', sent)
        return sent.capitalize()

