# Turtle Soup Boiler
A simple situation puzzle GPT based generator

## Usage
  ```
  from turtle_soup_boiler import TurtleSoupBoiler           # Import the package
  tsb = TurtleSoupBoiler(filename='good_story.pkl')         # Define the boiler object
  tsb.generate_by_input()                                   # Generate story by keyboard input a initial start
  tsb.generate_by_text()                                    # Generate story by input a string 
  tsb.save_story()                                          # Save all generated story into a .pkl file (default story.pkl)
  ```
