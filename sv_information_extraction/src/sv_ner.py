from collections import Counter
import nltk
from nltk import Tree
from nltk.chunk import RegexpParser
import re
import spacy
from spacy import displacy
from spacy.tokens import Span

# split the input doc into sentences
def split_sents(text):
    return nltk.sent_tokenize(text)

# returns True if a token contains an uppercase letter at any point
def contains_uppercase(name):
    cases = list(map(str.isupper, name)) # all the letters and cases

    count = sum(case == True for case in cases) # where an uppercase is
    if cases[0] == True or count > 0 or count == len(name):
        return True # if an uppercase was found
    return False

# Returns a list of tuples of words and their tags
# E.g., return format: [("word", "tag"), ...]
def tagged_sents(doc):
    # loop through words
    for index, token in enumerate(doc):
        upper = contains_uppercase(token.text)
        # if word is a noun, has an uppercase character and is not
        # the first word in the sentence
        if token.tag_.split("|")[0] == "NN" and upper and index != 0:
            token.tag_ = "UN" # assign updated noun tag
    return [(token.text, token.tag_.split("|")[0]) for token in doc]

# return the spacy assigned entities
def spacy_ents(doc):
    if doc.ents: # if it has entities
        return [(ent.label_, tagged_sents(ent)) for ent in doc.ents]
    return []

# return an NLTK Tree String representation of a sentence
def tree_string(tsents):
    s = "(S "
    for tsent in tsents: # loop through tagged sentence
        s = s + tsent[0] + "/" + tsent[1] + " " # split into token and POS tag
    return s[0:len(s)-1] + ")" # return the string and remove final space

# construct and return an NLTK Tree from spaCy entities
def tree(ne, s):
    if len(ne) > 0: # if more than one entity
        for label, e in ne: # for each label, entity
            f_ne = "(" + label #start string
            for i in e:
                f_ne = f_ne + " " + i[0] + "/" + i[1] # split entity and tag
            f_ne = f_ne + ")" # close off string
            ne_group = f_ne[f_ne.find(" ") + 1:-1]
            if s.find(ne_group) > 0: # construct required Tree string format
                s = s.replace(s[s.index(ne_group):s.index(ne_group) + len(ne_group)], f_ne)
    return nltk.tree.Tree.fromstring(s) # return as Tree

# extracts the labels from the tokens in an NLTK Tree
def tree_labels(tree):
    labels = ""
    for st in tree: # for each token in tree
        if hasattr(st, "label"): # if it is a named entity
            labels += st.label() + " " # get the label
        else:
            split = st.split("/") # split into token, POS tag

            if len(split) == 2: # split into two, so get POS tag
                labels += split[1] + " "
            else:
                labels += "RG" + " " # assume regular number
    return labels

# apply patterns to entities through NLTK
def v2_ents(labels, tsents):
    pattern = r"""ORG: {<PM>+<UN>*}"""
    if re.search("^TME\sVB\sPM", labels):
        pattern = r"""ORG: {<PM>}""" # LOC, LOC probably
        # i 1975 registers, but 1975 alone does not. PM RG
        # TME TME

    parser = RegexpParser(pattern)
    return parser.parse(tsents) # constructs a Tree and returns

# combines two NLTK Tree representations
def combine_trees(tree, tree2):
    ne_sent = "(S " # start Tree sentence
    last_element = ""
    for branch in tree: # loop through Tree 1
        if hasattr(branch, "label"):
            if not str(branch) in ne_sent: # if its a named entity and not already
                ne_sent += str(branch) + " " # in the list, add it to the list
        else:
            for branch2 in list(tree2): # loop through tree 2
                if hasattr(branch2, "label"): # named entity
                    if str(branch) in str(branch2) and str(branch2) not in ne_sent:
                        ne_sent += str(branch2) + " " # add if not there already
                else:
                    branch2 = str(branch2[0]) + "/" + str(branch2[1])
                    # regular token so breakdown
                    if str(branch) in str(branch2) and str(branch) not in ne_sent:
                        ne_sent += str(branch) + " " # check if the token is in list already
                    elif str(branch) in str(branch2) and str(branch) in ne_sent and str(branch) != last_element:
                        found = False
                        for ne in re.findall('\(.*?\)', ne_sent): # get anything between brackets
                            if str(branch) in ne and len(ne) > len(str(branch)) + 2:
                                # accomodate for brackets
                                found = True
                                
                        if found: # add to the list
                            last_element = str(branch)
                            ne_sent += str(branch) + " "
    return ne_sent[0:len(ne_sent)-1] + ")" # return the Tree String

# Returns a list of NLTK Tree objects for each sentence
# E.g., return format: [Tree1]
def nltk_ne_trees(text, nlp):
    sents = split_sents(text)
    trees = []
    for sent in sents:
        doc = nlp(sent)
        tsents = tagged_sents(doc)
        # get spaCy entities
        ne = spacy_ents(doc)
        # construst NLTK Tree using spaCy entities
        spacy_tree = tree(ne, tree_string(tsents))
        labels = tree_labels(spacy_tree) # extract labels
        # cosntruct a pattern-based Tree
        v2_tree = v2_ents(labels, tsents)
        # combine the two trees to get best of both systems
        new_tree = nltk.tree.Tree.fromstring(combine_trees(spacy_tree, v2_tree))
        trees.append(new_tree) # add tree to list
    return trees

# extracts NLTK entities from a Tree in a desired format
def format_nltk_ents(sent, nltk_ents):
    ents = []
    start = 0
    end = 0
    for branch in nltk_ents:# for each branch
        if hasattr(branch, "label"): # if it is a named entity or phrase
            nltk_ent = str(branch)[str(branch).find(" "):-1]
            split = nltk_ent.split("/") # get the ent and split into text and tag

            ent = ""
            for token in sent: # loop through each token in the sentence
                for word in split: # loop through each word
                    if token.text in word and word.count(token.text) == 1 and token.text not in ent:
                        if len(token.text) == 1 and word.index(token.text) in [0, 1]:
                            ent += token.text + " "
                        elif len(token.text) > 1:
                            ent += token.text + " " # normal word
                        # if the token is equal to the word and its not already
                        # in the entities list, then add it
            ent = ent[0:-1]
            if len(ent) > 0: # if entity string is not empty
                start = end
                end = start + len(ent.split(" ")) # append entity with start/end position
                ents.append((start, end, branch.label(), ent))
        else:
            start += 1
            end += 1 # increment
    return ents

# returns the complete NE list
def spacy_ne(text, nlp): # get the entities as an NLTK tree
    nltk_ents = nltk_ne_trees(text, nlp)
    
    spacy_ents = [] # loop through each sentence
    for sent_index, nltk_ents in enumerate(nltk_ents):
        sent = nlp(split_sents(text)[sent_index])
        sent.ents = list() # empty the spaCy entities
        # add the NLTK/pattern mix entities
        ents = format_nltk_ents(sent, nltk_ents)

        for ent in ents: # add to list
            spacy_ents.append((sent, ent[3] + " (" + ent[2] + ")"))
                
    return spacy_ents # return to user
