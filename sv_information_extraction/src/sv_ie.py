import nltk
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
import pandas as pd
import random
import spacy
from spacy import displacy
from spacy.matcher import Matcher
import src
from src.sb_corpus_reader import SBCorpusReader

# return a pre-tagged sentence, i.e. from a corpus
# as a regular string
def tagged_sent_as_str(sent):
    return TreebankWordDetokenizer().detokenize(sent)

# returns True if the verb is a compound verb
# i.e. if it depends on another phrase
def is_comp_verb(text, index):
    token = text[index]
    # for each token, check its dependency
    for sub_token in token.children:
        if sub_token.dep_ in ["ccomp", "xcomp"]:
            return True
    return False

    return verbs

# checks if a target dependency is in a subtree
def dep_is_in_tree(dep, tree):
    for token in tree: # loop through each token and check
        if token.dep_ in dep: # if its dependency matches
            return True
    return False

# matches a noun which is preceded by a modifier
def match_modified_noun(text, index):
    token = text[index]
    # extract the token and loop through its left tokens
    noun_phrase = ""
    for left in token.lefts: # if a token is a noun and has a modifier
        if left.pos_ == "NOUN" and left.dep_ == "nmod":
            noun_phrase += left.text + " " # add it to the phrase
    noun_phrase += token.text # add the original token
    return noun_phrase

# returns the subject part of phrase
# i.e., the head of the phrase
# takes a verb as an input and tries to find its subject
def get_subject_phrase(text, index):
    token = text[index]
    # determine whether the sentence follows Swedish V2 rule or not
    if not dep_is_in_tree(["nsubj"], token.lefts) and token.i != 0 and len(list(token.rights)) > 0:
        # v2 rule, so the word order is switched (Chapter 2 of the dissertation)
        noun_phrase = ""
        if dep_is_in_tree(["nsubj", "nsubj:pass"], token.rights): # if there is a subject dependency on right
            for token_right in token.rights: # loop through right hand side
                if token_right.pos_ in ["NOUN", "PRON", "PROPN"] and token_right.dep_ in ["nsubj", "nsubj:pass"]:
                    # noun found which modifies the verb
                    noun_phrase = match_modified_noun(text, token_right.i) + " " # get any modifiers
                    for child in token_right.rights: # loop through right hand side of noun
                        if child.pos_ in ["NOUN", "PROPN"] and child.dep_ in ["flat", "flat:name", "nmod", "nmod:poss"]:
                            noun_phrase += child.text + " " # if any additional modifiers are found, add
        elif token.nbor(1).pos_ in ["NOUN", "PRON", "PROPN"]:
            nbor = token.nbor(1) # if the neighbour to the right is a noun
            while nbor.pos_ in ["NOUN", "PROPN"] and len(list(nbor.rights)) > 0:
                noun_phrase += nbor.text + " "
                # loop and add
                temp = nbor
                nbor = temp.nbor(1)

        if len(noun_phrase) > 0 and noun_phrase[-1] == " ": # remove the trailing space
            return noun_phrase[0:len(noun_phrase) - 1]
    else: # not the V2 rule, normal structure
        subject_phrase = ""
        if token.pos_ == "VERB" and token.dep_ not in ["aux", "xcomp", "ccomp"]:
            for left in token.lefts: # if the current token is a proper verb
                if left.pos_ in ["NOUN", "PRON", "PROPN"] and left.dep_ == "nsubj":
                    if left.pos_ == "PRON": # go through left tokens and see if a subject is found
                        # just add it to the list
                        subject_phrase += left.text
                    elif left.dep_ == "aux": # if its an auxiliary verb, add before verb
                        verb_phrase += left.text + " "
                    else:
                        for left2 in left.lefts: # loop through left tokens
                            if left2.pos_ in ["NOUN", "PROPN"] and left2.dep_ in ["flat", "flat:name", "nmod", "nmod:poss"]:
                                subject_phrase += left2.text # found a modifier, so add
                        if left.nbor(1) == token: # if right hand is original token
                            if len(subject_phrase) > 0:
                                subject_phrase += " " + left.text # add it to the phrase
                            else:
                                subject_phrase += left.text
                        else:
                            if len(subject_phrase) > 0: # if subject phrase is not 0
                                subject_phrase += " " + left.text
                            else:
                                subject_phrase += left.text
                            for right2 in left.rights: # go through rights and find any modifiers
                                if right2.pos_ in ["NOUN", "PROPN"] and right2.dep_ in ["flat", "flat:name", "nmod", "nmod:poss"]:
                                    subject_phrase += " " + right2.text
        return subject_phrase
                    
    return ""

# returns a verb phrase which has auxiliaries, i.e. to want to sing
def get_verb_phrase(text, index):
    token = text[index]
    
    verb_phrase = ""
    for left in token.lefts: # loop through lefts
        if left.dep_ == "aux": # add auxiliary
            verb_phrase += left.text + " "
    verb_phrase += token.text

    return verb_phrase

# returns the object part of a phrase, i.e. the tail end
# also adds to the verb part
def get_object_phrase(text, verb_phrase, index):
    token = text[index] 

    object_phrase = ""
    for right in token.rights: # loop through right hand tokens
        if right.pos_ == "VERB" and right.dep_ in ["ccomp", "xcomp"]:
            mark_phrase = "" # if it is a verb and comp
            for child in right.children: # loop through dependent children of right
                if child.dep_ == "mark":
                    mark_phrase += child.text + " " # if infinitive mark, add 
                elif child.dep_ in ["obj", "obl"] and len(object_phrase) == 0:
                    if child.nbor(-1).pos_ == "ADP" and child.nbor(-1).dep_ == "case":
                        verb_phrase += " " + child.nbor(-1).text # or if an adposition object, also add
                    object_phrase += child.text
                elif child.pos_ == "ADV" and child.dep_ == "advmod":
                    verb_phrase += child.text + " " # add an adverbial modifier
            verb_phrase += " " + mark_phrase + right.text
        elif right.pos_ in ["NOUN", "PROPN", "PRON"] and right.dep_ in ["obj", "obl"] and len(object_phrase) == 0:
            if not is_comp_verb(text, token.i) and token.nbor(1) == right and right.text in ["mig", "dig", "sig"]:
                object_phrase += right.text # if right hand is a noun and not mig, dig or sig
            else:
                for child in token.children:
                    if child.pos_ == "VERB":
                        for child2 in child.children: # loop through children of the verb
                            if child2.pos_ in ["NOUN", "PRON", "PROPN"] and child2.dep_ == "obl":
                                if len(list(child2.lefts)) > 0 and child2.nbor(-1).pos_ == "ADP" and child2.nbor(-1).dep_ == "case":
                                    verb_phrase += " " + child2.nbor(-1).text
                                object_phrase += child2.text # add to the phrase
                if len(object_phrase) == 0:
                    if right.nbor(-1).pos_ == "ADP" and right.nbor(-1).dep_ == "case":
                        verb_phrase += " " + right.nbor(-1).text # add the left hand neighbour
                    object_phrase += right.text
        elif right.pos_ == "ADV" and right.dep_ == "advmod" and not (not dep_is_in_tree(["nsubj"], token.lefts) and token.i != 0 and len(list(token.rights)) > 0):
            verb_phrase += " " + right.text

    return verb_phrase, object_phrase

# matches a verb phrase, such as the boy jumped on the road
def match_verb_phrases(text):
    matches = []
    for token in text: # for each token in the sentence
        if token.pos_ == "VERB" and token.dep_ != "amod": 
            phrase = []
            # if its a verb and not modifyin
            subject_phrase = get_subject_phrase(text, token.i) # get subject phrase
            phrases = get_object_phrase(text, get_verb_phrase(text, token.i), token.i)
            verb_phrase = phrases[0] # get object and verb phrase
            object_phrase = phrases[1]

            # if each phrase is not 0, add it to the list
            if len(subject_phrase) > 0:
                phrase.append(subject_phrase)

            if len(verb_phrase) > 0:
                phrase.append(verb_phrase)

            if len(object_phrase) > 0:
                phrase.append(object_phrase)

            # if the phrase has all three constituents, add
            if len(phrase) == 3 and subject_phrase != object_phrase:
                matches.append(phrase)
    return matches

# returns nouns and their descriptors, i.e.
# the red car, etc.
def match_descriptive_nouns(text):
    matches = []

    for token in text:
        phrase = ""
        # not grammatically sound for proper nouns or pronouns
        if token.pos_ == "NOUN" and token.dep_ in ["obj", "nsubj", "nsubj:pass", "csubj", "obl", "ROOT"]:
            for sub_token in token.children: # if there is an adjective and it is a modifier
                if sub_token.pos_ == "ADJ" and sub_token.dep_ == "amod":
                    phrase += sub_token.text + " "
                    # add it to the phrase
            if len(phrase) != 0:
                phrase += token.text # adjectives precede noun

        if len(phrase) != 0:
            matches.append(phrase)

    return matches

# attaches any noun modifiers to the noun
def modify_noun(text, index):
    token = text[index]

    noun_phrase = "" # for each token, check if a modifier exists
    for sub_token in token.children:
        if sub_token.dep_ in ["amod", "compound"]:
            noun_phrase += sub_token.text + " " # attach to phrase
    noun_phrase += token.text

    return noun_phrase

# matches any preposition-bound nouns, i.e. i huset (in the house)
def match_preposition_nouns(text):
    matches = []
    # for each token in an input doc
    for token in text:
        if token.pos_ == "ADP": # if POS is adposition
            phrase = ""
            # find the head of the token and see if it is a noun
            if token.head.pos_ in ["NOUN", "PROPN"]:
                head = token.head

                # if the head of the head is a noun
                if head.head.pos_ == "NOUN" and head.head != head:
                    modified_noun_phrase = modify_noun(text, head.head.i)
                    if len(modified_noun_phrase) > 0: # find the modifiers
                        phrase += modified_noun_phrase + " "
                    else: # and add to the phrase
                        phrase += head.head.text + " "

                    phrase += token.text + " "

                    modified_noun_phrase = modify_noun(text, head.i)
                    if len(modified_noun_phrase) > 0:
                        phrase += modified_noun_phrase
                    else:
                        phrase += head.text
                    # append the phrase to the match list
                    matches.append(phrase)

    return matches # return match list

# return the input key function as a string to be displayed
def func_as_str(func): # remove _ and replace with " "
    return (func[0].upper() + func[1:-1]).replace("_", " ")

# return the results as a pandas DataFrame object to be represented
def results_as_pandas_df(sents, func, nlp, is_sample):
    func_str = func_as_str(func)
    # construct df with following columns
    df = pd.DataFrame(columns=["ID", "Sentence", "Length", func_str])
    row_list = []
    # for each sentence
    for index, sent in enumerate(sents):
        sent_str = sent
        if is_sample: # if its a sample, then get as a string
            sent_str = tagged_sent_as_str(sent)
        # if the function string is one of the following
        if func == "descriptive_nouns":
            row_list.append({"ID":index, "Sentence":sent_str, "Length":len(sent), func_str:match_descriptive_nouns(nlp(sent_str))})
        elif func == "preposition_nouns":
            row_list.append({"ID":index, "Sentence":sent_str, "Length":len(sent), func_str:match_preposition_nouns(nlp(sent_str))})
        elif func == "verb_phrases":
            row_list.append({"ID":index, "Sentence":sent_str, "Length":len(sent), func_str:match_verb_phrases(nlp(sent_str))})
        # perform the appropriate function and add the result to df
    return pd.DataFrame(row_list)

# calculates the percentage of correctly matched
# sentences for a pattern
def output_percentage(df, col):
    result = 0
    
    for c in df[col]: # for each column
        if len(c) != 0: # if not False
            result += 1
    
    percentage = result / len(df)
    percentage *= 100 # calculate percentage
    
    return round(percentage, 1)

# sets up pandas
def setup_pandas():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1820)

# returns a pandas DataFrame object of all the information
def pandas_df(is_sample, sents, key, nlp):
    df = results_as_pandas_df(sents, key, nlp, is_sample)
    dis_list = [] # construct dictionaries

    percentage = output_percentage(df, func_as_str(key))
    if percentage != 0:
        print("Percentage: ", output_percentage(df, func_as_str(key)), "\n")
    else:
        print("No matches were found")
    
    df_show = pd.DataFrame(columns=df.columns)
    # output the match percentage and construct as df
    for row in range(len(df)):
        if len(df.loc[row, func_as_str(key)]) != 0:
            df_show = df_show.append(df.loc[row,:])
    # add to the df_show object to be represented
    df_show.reset_index(inplace = True)
    df_show.drop("index", axis = 1, inplace = True)

    verb_dict = dict()
    dis_dict = dict()

    for i in range(len(df_show)):
        index = df_show.loc[i, "ID"]
        sentence = df_show.loc[i, "Sentence"]
        length = df_show.loc[i, "Length"]
        output = df_show.loc[i, func_as_str(key)]
        # get the key details to represent as a df
        for sent in output:
            if key == "verb_phrases":
                # separate subject, verb and object
                subj, verb, obj = sent[0], sent[1], sent[2]
            
                # append to list, along with the sentence
                dis_dict = {"ID":index, 'Sentence':sentence,'Length':length,'Subject':subj,'Verb':verb,'Object':obj}
                dis_list.append(dis_dict)
            else: # normal output through pandas, i.e. no separation of constituents
                dis_dict = {"ID":index, "Sentence":sentence, "Length":length, func_as_str(key):sent}
                dis_list.append(dis_dict)
                
    return pd.DataFrame(dis_list)

def split_sents(text):
    return nltk.sent_tokenize(text)

# extract information using an input dataset
def extract_info(text, key, nlp):
    setup_pandas()

    text = split_sents(text) # split into sentences
    # output to the user
    print("")

    df = pandas_df(False, text, key, nlp)
    if len(df) > 0:
        print(df)

# extract information using a sample dataset (requires different input)
def extract_info_sampleset(sample_data, key, nlp):
    setup_pandas()
    # print the output to the user
    print("")
    
    df = pandas_df(True, sample_data, key, nlp)
    if len(df) > 0:
        print(df)
