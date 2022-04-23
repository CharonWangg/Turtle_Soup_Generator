from .utils import *
import pickle


class TurtleSoupBoiler:
    def __init__(self, num_sent=5, p_sample=0.6, sample_step=2, filename=None):
        configure_openai()
        self.__dict__.update(locals())
        self.story = []
        if filename is None:
            self.filename = 'story.pkl'

    # Generate a story by input a sentence and store it in self.story
    def generate_by_input(self, verbose=False):
        print('>Please input the first sentence of your story:')
        sentence = input()
        story = generate_story(sentence, self.num_sent, self.p_sample, self.sample_step, verbose)
        self.story.append(story)
        print('>Generate story successfully!')

    # Generate a story by string and store it in self.story
    def generate_by_text(self, text, verbose=False):
        story = generate_story(text, self.num_sent, self.p_sample, self.sample_step, verbose)
        self.story.append(story)
        print('>Generate story successfully!')

    # Save the story to a csv file
    def save_story(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.story, f)
            print('>Save story successfully!')


