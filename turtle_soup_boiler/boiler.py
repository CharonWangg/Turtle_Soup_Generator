from .utils import *
import pickle


class TurtleSoupBoiler:
    def __init__(self, filename=None):
        configure_openai()
        self.story = []
        if not filename:
            self.filename = 'story.pkl'

    # Generate a story by input a sentence and store it in self.story
    def generate_by_input(self):
        print('>Please input the first sentence of your story:')
        sentence = input()
        story = generate_story(sentence)
        self.story.append(story)
        print('>Generate story successfully!')

    # Generate a story by string and store it in self.story
    def generate_by_text(self, text):
        story = generate_story(text)
        self.story.append(story)
        print('>Generate story successfully!')

    # Save the story to a csv file
    def save_story(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.story, f)
            print('>Save story successfully!')


