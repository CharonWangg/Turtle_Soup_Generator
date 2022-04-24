from boiler import TurtleSoupBoiler           # Import the package
tsb = TurtleSoupBoiler(
    num_sent=12, 
    p_sample=0.7, 
    sample_step=3,   
    filename='good_story.pkl',
    verbose=True
)
tsb.generate_by_input()                      # Generate story by keyboard input a initial start
# last_sent = tsb.story[-1].split('. ')[-1]
# print(tsb.story[-1].split('. '))
# tsb.generate_by_text(last_sent, verbose=False)                       # Generate story by input a string 
# tsb.save_story()                                          # Save all generated story into a .pkl file (default story.pkl)
# print(tsb.story)