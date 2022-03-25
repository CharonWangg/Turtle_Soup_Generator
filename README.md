# Turtle Soup Generator
A Situation Puzzle Generator | Final Project for CIS 700 Interactive Fiction Generation

Project Description
-------------------

A situation puzzle generator where the user is given the start and end of a story and is asked to guess the whole story. The user can query if a situation has happened and our model will return True or False as a hint. 

Methods:
--------

- Two models & Connecting Components
    - Generative model 
      - Input: start and end of the stories 
      - loss function: NLL
      - Evaluation: Story cloze test (?)
      - Implementation: TBD
    - QA model 
      - A simple classification model, just checking if the user’s guess exists in the steps (compare with: item lookup homework). 
      - Return only Yes/No to users, 
      - Implementation: TBD (likely GPT family models)
      - (optional) return additional hint in the future plan.
    - Connecting Generative model and QA model 

Data:
-----

- [Kith Situation Puzzle](https://kith.org/logos/things/sitpuz/situations.html)
    - The amount of data available is ~100 instances of situation puzzles. We have also found chinese versions available (which could be backup data if we need)

Related Work
------------

- Story Generation
    
- QA Model

Readling List
-------------
- [Automated Story Generation as Question-Answering](https://arxiv.org/pdf/2112.03808.pdf) 
    - Our proposed story generation system starts with sentences encapsulating the final event of the story and generates backward from a given ending.
- [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://arxiv.org/pdf/1905.10044.pdf)    
    - We find that transferring from entailment data is more effective than transferring from paraphrase or extractive QA data, and that it, surprisingly, continues to be very beneficial even when starting from massive pre-trained language models such as BERT.
