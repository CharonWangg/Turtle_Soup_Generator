from boiler import TurtleSoupBoiler

tsb = TurtleSoupBoiler(sample_step=1, num_sent=5, verbose=True)
init = "A man lets go of a bowling ball."
tsb.generate_by_input()
tsb.save_story()