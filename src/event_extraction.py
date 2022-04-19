'''
    Description: Extraction events (subject, verb, object, modifier) from a given sentence
    Reference: https://github.com/lara-martin/ASTER/blob/b74786e84c081096f518fea67ce1ea09ee0f9a47/Pruning%2BSplitting/eventmakerTryServer.py
    Dependency: (see setup.sh)
        - spacy
        - stanza
        - neuralcoref
        - nltk (wordnet, verbnet, lesk)
        - numpy
'''

################ Setup ################
# spacy
import spacy
import spacy.cli
try:
    core = spacy.load('en_core_web_lg')
    print('[spacy] No download needed!')
except:
    print('[spacy] Downloading file: [en_core_web_lg]')
    spacy.cli.download("en_core_web_lg")
    core = spacy.load('en_core_web_lg')
import neuralcoref
neuralcoref.add_to_pipe(core)

# stanza
import stanza
import json
try:
    nlp = stanza.Pipeline('en', processors='pos, lemma, tokenize, ner, depparse, mwt, constituency')
    print('[stanza] No download needed!')
except:
    stanza.download('en')
    print('[stanza] Downloading file: [en]')
    nlp = stanza.Pipeline('en', processors='pos, lemma, tokenize, ner, depparse, mwt, constituency')

# nltk
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
try:
    wn.synsets('dog')
    print('[nltk] No download needed for wordnet!')
except:
    print('[nltk] Downloading file: [wordnet]')
    nltk.download('wordnet')

from nltk.corpus import verbnet
try:
    verbnet.classids('have')
    print('[nltk] No download needed for verbnet!')
except:
    print('[nltk] Downloading file: [verbnet]')
    nltk.download('verbnet')

# system utils
from collections import defaultdict
from numpy.random import choice

################ EventMaker ################
class EventMaker:
    
    def __init__(self, sentence):
        self.nouns = defaultdict(list)
        self.verbs = defaultdict(list)
        self.sentence = sentence
        self.generalized_events = []
        self.original_events = []

    def getEvent(self):
        original_sent = self.sentence.strip()
        
        # parsing
        parse = nlp(self.sentence)
        d = json.loads(str(parse))
        all_tokens = d[0]
        ner_dict2 = {}
        for token in all_tokens:
            ner_dict2[token["text"]] = token["ner"]


        for sent_num, sentence in enumerate(parse.sentences): # for each sentence in the entire input
            tokens = defaultdict(list) 

            for token in all_tokens: 
                tokens[token["text"]] = [token["lemma"], token["xpos"], ner_dict2[token["text"]], token["id"]] # each word in the dictionary has a list of [lemma, POS, NER]
            # print('tokens:', tokens)
            deps = sentence.dependencies # retrieve the dependencies
            named_entities = []
            verbs = []
            subjects = []
            modifiers = []
            objects = []
            pos = {}
            pos["EmptyParameter"] = "None"
            chainMods = {} # chaining of mods
            index = defaultdict(list)  #for identifying part-of-speech
            index["EmptyParameter"] = -1
            # create events
            parsed_deps = []
            for dep in deps:
                #subject
                d = {}
                gov_, rel_, dep_ = dep
                d['dep'] = rel_
                d['governorGloss'] = gov_.text
                d['governor'] = gov_.id
                d['dependentGloss'] = dep_.text
                d['dependent'] = dep_.id
                parsed_deps.append(d)

            deps = parsed_deps
            for d in deps:
                if len(tokens[d["dependentGloss"]]) == 0:
                    continue
                if 'nsubj' in d["dep"] and "RB" not in tokens[d["dependentGloss"]][1]: #adjective? #"csubj" identifies a lot of things wrong 
                    #print(tokens[d["dependentGloss"]][1])
                    if d["governorGloss"] not in verbs:
                        #create new event
                        if not "VB" in tokens[d["governorGloss"]][1]: continue
                        verbs.append(d["governorGloss"])
                        index[d["governorGloss"]] = d["governor"] #adding index
                        subjects.append(d["dependentGloss"])
                        index[d["dependentGloss"]] = d["dependent"] #adding index to subject
                        pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
                        pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                        modifiers.append('EmptyParameter')
                        objects.append('EmptyParameter')
                    elif d["governorGloss"] in verbs:
                        if subjects[verbs.index(d["governorGloss"])] == "EmptyParameter": # if verb alrady exist 
                            subjects[verbs.index(d["governorGloss"])] = d["dependentGloss"]
                            index[d["dependentGloss"]] = d["dependent"]
                        else:
                            subjects.append(d["dependentGloss"])
                            index[d["dependentGloss"]] = d["dependent"]
                            verbs.append(d["governorGloss"])
                            index[d["governorGloss"]] = d["governor"]
                            modifiers.append('EmptyParameter')
                            objects.append('EmptyParameter')
                        pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
                        pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                    elif d["dependentGloss"] in subjects: # one subject multiple verbs
                        verbs[subjects.index(d["dependentGloss"])] = d["governorGloss"]
                        index[d["governorGloss"]] = d["governor"]
                        pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
                        pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                else: #check to see if we have a subject filled ??
                    if len(subjects) >1:
                        if subjects[-1] == "EmptyParameter":
                            subjects[-1] = subjects[-2]
                    #conjunction of verbs
                    if 'conj' in d["dep"] and 'VB' in tokens[d["dependentGloss"]][1]:
                        if d["dependentGloss"] not in verbs:
                            verbs.append(d["dependentGloss"])
                            pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                            index[d["dependentGloss"]] = d["dependent"]
                            subjects.append('EmptyParameter')
                            modifiers.append('EmptyParameter')
                            objects.append('EmptyParameter')
                    #conjunction of subjects
                    elif 'conj' in d["dep"] and d["governorGloss"] in subjects: # governor and dependent are both subj. e.g. Amy and Sheldon
                        loc = subjects.index(d["governorGloss"])
                        verb = verbs[loc] #verb already exist. question: should the verb have the same Part-of-Speech tag?
                        subjects.append(d["dependentGloss"])
                        index[d["dependentGloss"]] = d["dependent"]
                        verbs.append(verb)
                        modifiers.append('EmptyParameter')
                        objects.append('EmptyParameter')
                        pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                    elif 'conj' in d["dep"] and d["governorGloss"] in objects:
                        loc = objects.index(d["governorGloss"])
                        match_verb = verbs[loc] #??? is it a correct way to retrieve the verb?
                        #print(match_verb)
                        temp_verbs = copy.deepcopy(verbs)
                        for i, verb in enumerate(temp_verbs):
                            if match_verb == verb: # what if the verb appears more than one times?
                                subjects.append(subjects[i]) 
                                verbs.append(verb)
                                modifiers.append('EmptyParameter')
                                objects.append(d["dependentGloss"])
                                index[d["dependentGloss"]] = d["dependent"]
                        pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]			
                    # case 1: obj
                    elif 'dobj' in d["dep"] or 'xcomp' == d["dep"]:  #?? 'xcomp' is a little bit tricky
                        if d["governorGloss"] in verbs:
                            #modify that object
                            pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                            for i, verb in reversed(list(enumerate(verbs))):
                                if verb == d["governorGloss"] and objects[i] == "EmptyParameter":
                                    objects[i] = d["dependentGloss"]
                                    index[d["dependentGloss"]] = d["dependent"]
                    # case 2: nmod
                    elif ('nmod' in d["dep"] or 'ccomp' in d["dep"] or 'iobj' in d["dep"] or 'dep' in d["dep"]) and 'NN' in tokens[d["dependentGloss"]][1]:
                        if d["governorGloss"] in verbs: # how about PRP?
                            #modify that modifier
                            for i, verb in reversed(list(enumerate(verbs))):
                                if verb == d["governorGloss"] and modifiers[i] == "EmptyParameter":
                                    modifiers[i] = d["dependentGloss"]
                                    index[d["dependentGloss"]] = d["dependent"]
                            pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                        elif d["governorGloss"] in chainMods: # is not used actually
                            v = chainMods[d["governorGloss"]]
                            if v in verbs:
                                modifiers[verbs.index(v)] = d["dependentGloss"]
                                index[d["dependentGloss"]] = d["dependent"]
                                pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                    # PRP			
                    elif ('nmod' in d["dep"] or 'ccomp' in d["dep"] or 'iobj' in d["dep"] or 'dep' in d["dep"]) and 'PRP' in tokens[d["dependentGloss"]][1]:
                        if d["governorGloss"] in verbs: # how about PRP?
                            #modify that modifier
                            for i, verb in reversed(list(enumerate(verbs))):
                                if verb == d["governorGloss"] and modifiers[i] == "EmptyParameter":
                                    modifiers[i] = d["dependentGloss"]
                                    index[d["dependentGloss"]] = d["dependent"]
                            pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
                        elif d["governorGloss"] in chainMods: # is not used actually
                            v = chainMods[d["governorGloss"]]
                            if v in verbs:
                                modifiers[verbs.index(v)] = d["dependentGloss"]
                                index[d["dependentGloss"]] = d["dependent"]
                                pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]

            # generalize the words and store them in instance variables
            for (a,b,c,d) in zip(subjects, verbs, objects, modifiers):
                pos1 = "None"
                pos2 = "None"
                pos3 = "None"
                pos4 = "None"
                poslabel = "None"
                # print((a,b,c,d))
                num = 0
                if a != 'EmptyParameter':
                    if index[a] == tokens[a][-1]: #adding part-of-speech
                        pos1 = tokens[a][1]
                    a1, named_entities = self.generalize_noun(a, tokens, named_entities, original_sent)
                    if "<NE>" in a1:
                        self.nouns["<NE>"].append(tokens[a][0])
                    elif a1 == a:
                        a1 = self.generalize_verb(a, tokens) #changed line
                        self.verbs[a1].append(tokens[a][0])
                    else:
                        self.nouns[a1].append(tokens[a][0])
                else:
                    a1 = a
                if b != 'EmptyParameter':
                    if index[b] == tokens[b][-1]: #may have issue in looping
                        pos2 = tokens[b][1]
                    b1 = self.generalize_verb(b, tokens) #changed line
                    self.verbs[b1].append(tokens[b][0])
                else:
                    b1 = b
                if c != 'EmptyParameter':
                    if index[c] == tokens[c][-1]:
                        pos3 = tokens[c][1]
                    c1, named_entities = self.generalize_noun(c, tokens, named_entities, original_sent)
                    if "<NE>" in c1:
                        self.nouns["<NE>"].append(tokens[c][0])
                    elif c1 == c:
                        c1 = self.generalize_verb(c, tokens) #changed line
                        self.verbs[c1].append(tokens[c][0])
                    else:
                        self.nouns[c1].append(tokens[c][0])
                else:
                    c1 = c
                if d == 'EmptyParameter': #adding new lines start. 
                    label = 'EmptyParameter'
                else:
                    label = 'None' #change from Exist
                    for dep in deps:
                        if b == dep["governorGloss"] and d == dep["dependentGloss"] and "nmod" in dep["dep"]:
                            if ":" in dep["dep"]:
                                label = dep["dep"].split(":")[1]  # shoud this line be added??
                            num = dep['dependent']
                            #print(dep)
                            #print("number:")
                            #print(num)
                            #print(type(num))
                    for dep in deps: # how about obl dependency?
                        if b == dep["governorGloss"] and d == dep["dependentGloss"] and "obl" in dep["dep"]:
                            if ":" in dep["dep"]:
                                label = dep["dep"].split(":")[1]
                            num = dep['dependent']
                            #print(dep)

                    for dep in deps:
                        if "case" in dep["dep"] and d == dep["governorGloss"] and num == dep['governor']: # #what if modifier is related to multiple labels?
                            label = dep["dependentGloss"]	#adding new lines end
                            index[label] = dep["dependent"]
                            #print("label:")
                            #print(num)
                            #print(dep['governor'])
                            #print(type(dep['governor']))
                if d != 'EmptyParameter':
                    if index[d] == tokens[d][-1]:
                        pos4 = tokens[d][1]
                    d1, named_entities = self.generalize_noun(d, tokens, named_entities, original_sent)
                    if "<NE>" in d1:
                        self.nouns["<NE>"].append(tokens[d][0])
                    else:
                        self.nouns[d1].append(tokens[d][0])
                else:
                    d1 = d	
                if label != "EmtpyParameter" and label != "None":
                    if len(tokens[label]) == 4:
                        #print(tokens[label])
                        #print(index[label])
                        if tokens[label][-1] == index[label]:
                            poslabel = tokens[label][1]

                #self.events.append([a1,b1,c1,label,d1])
                self.generalized_events.append([a1,b1,c1,d1])
                
                lemmatized = []
                for w in [a, b, c, d]:
                    if len(tokens[w]) > 0:
                        lemmatized.append(tokens[w][0])
                    else:
                        lemmatized.append(w)
                self.original_events.append(lemmatized)

    def generalize_noun(self, word, tokens, named_entities, original_sent):
        # This function is to support getEvent functions. Tokens have specific format(lemma, pos, ner)
        lemma = tokens[word][0]
        pos = tokens[word][1]
        ner = tokens[word][2]
        resultString = ""

        if ner != "O": # output of Stanford NER: default values is "O"
            if ner == "PERSON":
                if word not in named_entities: # named_entities is a list to store the names of people
                    named_entities.append(word)
                resultString = "<NE>"+str(named_entities.index(word))
            else:
                resultString = ner 
        else:
            word = lemma
            if "NN" in pos: # and adjective? #changed from only "NN"
                resultString = self.lookupNoun(word, pos, original_sent) # get the word's ancestor
            elif "JJ" in pos:
                resultString = self.lookupAdj(word, pos, original_sent)
            elif "PRP" in pos:
                if word == "he" or word == "him":
                    resultString = "Synset('male.n.02')"
                elif word == "she" or word == "her":
                    resultString = "Synset('female.n.02')"
                elif word == "I" or word == "me" or word == "we" or word == "us":
                    resultString = "Synset('person.n.01')"
                elif word == "they" or word == "them":
                    resultString = "Synset('physical_entity.n.01')"
                else:
                    resultString = "Synset('entity.n.01')" 
            else:
                resultString = word
        return resultString, named_entities


    def generalize_verb(self, word, tokens):
        # This function is to support getEvent functions. Tokens have specific format:tokens[word] = [lemma, POS, NER]
        word = tokens[word][0]
        if word == "have": return "own-100" 

        classids = verbnet.classids(word)
        if len(classids) > 0:
            #return choice based on weight of number of members
            mems = []
            for classid in classids:
                vnclass = verbnet.vnclass(classid)
                num = len(list(vnclass.findall('MEMBERS/MEMBER')))
                mems.append(num)
            mem_count = mems
            mems = [x/float(sum(mem_count)) for x in mems]
            return str(choice(classids, 1, p=mems)[0])
        else:
            return word
        
    def lookupNoun(self, word, pos, original_sent):
        #print(word, pos)
        # This is a function that supports generalize_noun function 
        if len(wn.synsets(word)) > 0:
            #word1 = lesk(original_sent.split(), word, pos='n')  #word1 is the first synonym of word
            #print(word1)
            return str(lesk(original_sent.split(), word, pos='n'))
        else:
            return word.lower()
    def lookupAdj(self,word, pos, original_sent):
        #print(word, pos)
        # This is a function that supports generalize_noun function 
        if len(wn.synsets(word)) > 0:
            #word1 = lesk(original_sent.split(), word, pos='n')  #word1 is the first synonym of word
            #print(word1)
            return str(lesk(original_sent.split(), word, pos='a'))
        else:
            return word.lower()	

import time
from tqdm import tqdm
################ Use Sample ################
if __name__ == '__main__':
    ########### Parse all evaluation ###########
    # file = open('./data/evaluation.txt', 'r')
    # sentences = []
    # for line in file.readlines():
    #     sentences.append(line.strip().replace('\n', ''))

    # print('Count of sentences:', len(sentences))
    # start = time.time()
    # event_makers = [EventMaker(sent) for sent in sentences]
    # for em in tqdm(event_makers):
    #     em.getEvent()
    # end = time.time()
    # print('Total time:', end - start)

    ########### Example usage ###########
    sentences = ['They try transferring shield power to the engines without effect.']
    for sent in tqdm(sentences):
        em = EventMaker(sent)
        em.getEvent()
        print('[Original sentence]:', em.sentence)
        print('[Original events]:', em.original_events)
        print('[Generalized events]:', em.generalized_events)
        print()