import json
from boiler import TurtleSoupBoiler           # Import the package

tsb = TurtleSoupBoiler(
    num_sent=12, 
    p_sample=0.7, 
    sample_step=3,   
    filename='good_story.pkl',
    verbose=True
)

######## Option 1: Test by single sentence ########
# tsb.generate_by_input()

######## Option 2: Test by sentence from config/boiler.json ########
first_sent = json.load(open('./config/boiler.json'))["first_sent"]
tsb.generate_by_text(first_sent)

######## Option 3: Test from list of sentences ########
