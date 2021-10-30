from collections import Counter
import lemmy
import nltk
from nltk.corpus import stopwords
import spacy

# splits an input doc into its component sentences
def split_sents(text):
    return nltk.sent_tokenize(text)

# splits a doc into sentences and prints them out
def print_sents(text):
    for sent in split_sents(text): # loop through
        print("\n" + sent)         # sentences

# prints out the syntactic info for an input doc
def print_syntactic_info(doc, nlp):
    for sent in split_sents(doc.text):
        print("") # loop through sentences
        for token in nlp(sent): # for each token, print syntactic info
            print(f"{token.text:{15}} {token.dep_:{20}} {token.pos_:{20}} {token.tag_:{20}}")

# removes the stopwords from an input text
def remove_stopwords(text, nlp): # add to list if not in stopwords list
    words = [word for word in text.split() if word not in stopwords.words("swedish")]
    return nlp(" ".join(words)) # return the stopwords

# prints the word frequency for 5 most common tokens
def print_word_frequency_list(doc, nlp):
    no_stopwords = remove_stopwords(doc.text, nlp) # remove stopwords as
    # these are particularly common and not of importance
    words = [token for token in no_stopwords if not token.is_punct]
    freq = Counter(words) # add to Counter object if not punctuation
    # output the 5 most common tokens
    print("\n" + str(freq.most_common(5)))

# prints the POS tag frequency list
def print_pos_frequency_list(doc):
    POS_count = doc.count_by(spacy.attrs.POS)
    print("") # count number of each POS tag
    for i, v in sorted(POS_count.items()): # output to user
        print(f"{doc.vocab[i].text:{5}}: {v}")

# print the tokens in an input doc
def print_tokens(doc, nlp):
    for sent in split_sents(doc.text):
        print("") # for each sentence, loop through tokens
        for token in nlp(sent): # output token
            print(token, token.idx)

# print the stopwords in an input doc
def print_stopwords(doc):
    stopwords_list = []
    for token in doc: #for each token, add stopword to list
        if token.text in stopwords.words("swedish") and not token.text in stopwords_list:
            stopwords_list.append(token.text)
    print("")
    for stopword in stopwords_list:
        print(stopword) # output to user

# prints the dependency skeleton for an input doc
def print_dependency_skeleton(doc, nlp):
    for sent in split_sents(doc.text):
        print("") # for each sentence, output the token and
        # morphological/syntactic information
        for token in nlp(sent):
            print(f"{token.text:{15}} {token.dep_:{20}} {token.head.text:{20}}")

# prints the lemmatised form of tokens in an input doc
def print_lemmatise_doc(doc, nlp):
    lemmatiser = lemmy.load("sv")
    # load up lemmy and loop through sentences
    for sent in split_sents(doc.text):
        print("")
        for token in nlp(sent): # loop through tokens and print lemmas
            lemma = lemmatiser.lemmatize(token.pos_, token.text)[0]
            print(f"{token.text:{15}} {lemma:{15}}")
