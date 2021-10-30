import nltk
import os
from os import path
import random
import spacy
import src
from src import sv_ie, sv_ner, sv_parser
from src.sb_corpus_reader import SBCorpusReader
import sys
from sys import argv

docs = []
current_doc = 0
nlp = None

# the general command-line commands
general_commands = ["change_doc", "print_docs", "print_doc",
                    "export_input_data", "export_current_doc", "help", "exit"]

# the parser specific commands
parser_commands = ["print_sentences", "syntactic_info", "POS_frequency_list",
                   "word_frequency_list", "print_tokens", "print_stopwords",
                   "print_dependencies", "print_lemmas"]

# the NER commands
ner_commands = ["nltk_entity_trees", "list_named_entities"]

# the IE commands
ie_commands = ["descriptive_nouns", "preposition_nouns", "verb_phrases"]

unintentional_sample_data = False

# clears the workspace/screen
def cls():
    os.system("cls" if os.name == "nt" else "clear")

# returns the current directory path
def get_current_dir():
    return path.dirname(path.realpath(__file__))

# get the directory for the input data
def get_input_data_dir():
    return path.join(get_current_dir(), "input_data")

# get the directory for the training data
def get_training_data_dir():
    return path.join(get_current_dir(), "training_data")

# get the directory for the trained models
def get_models_dir():
    return path.join(get_current_dir(), "models")

def get_output_data_dir():
    return path.join(get_current_dir(), "output_data")

# display help information to the user
def help():
    print("\nThis program can be run using the following format:")
    print("\tpython sv_information_extraction command file.txt")
    print("\nAvailable commands are: train, parse, ner, and ie.")
    print("\nFiles are located in:")
    print("\t", str(get_input_data_dir())[0:-1])
    print("\nSpecifying --sample in place of file.txt will use sample data instead.")

# return a sample set from SUC 3.0
def sample_training_set(size):
    if path.isfile(path.join(get_training_data_dir(), "suc3.xml")):
        suc = SBCorpusReader(path.join(get_training_data_dir(), "suc3.xml"))
        start = random.randint(0, 10000) # to avoid performance drop
        return suc.sents()[start:start + size]
    return []

# load the file or sample data
def load_file(uri):
    if uri != "":
        f = open(uri, "r", encoding = "utf-8")
        for line in f: # read file into list
            docs.append(line.strip())
        f.close()
    # empty file or chosen sample data
    if len(docs) == 0 or argv[2].lower() == "--sample":
        global unintentional_sample_data
        if len(docs) == 0 and argv[2].lower() != "--sample":
            unintentional_sample_data = True
            # file was empty, so give sample data

        data = sample_training_set(200)
        if len(data) > 0:
            for sample in data:
                docs.append(sv_ie.tagged_sent_as_str(sample))
        else:
            print("A relevant corpus, such as SUC-3, is required to use this tool.")
            exit()

# print the currently installed models
# there is sv_model_xpos by default
def print_models():
    mdir = get_models_dir() # get the directory
    models = os.listdir(mdir)
    filtered_models = []
    for model in models:
        if not model.startswith('.'):
            filtered_models.append(model)

    # display the list of models
    print("="*10, "Models", "="*10)
    for i, model in enumerate(filtered_models):
        print("[" + str(i) + "]", model)
    return filtered_models

# lets the user choose a model to use
def choose_model(models):
    global nlp

    loop = True
    while loop:
        try: # loop while the user is choosing
            choice = int(input("\nEnter the index of the model to use: "))

            if choice >= 0 and choice < len(models):
                model_path = path.join(get_models_dir(), models[choice]) # define model path
                if path.exists(path.join(model_path, "PATH.txt")): # ensure PATH.txt exists
                    with open(path.join(model_path, "PATH.txt"), "r", encoding="utf-8") as f:
                        model_path += path.join(model_path, f.read())
                        
                    nlp = spacy.load(model_path) # load selected model
                    cls() # clear the screen
                    print("Loaded model:", models[choice], "\n")
                    loop = False # display the loaded model and end loop
                else:
                    print("\nThe model cannot be located as its missing its PATH.txt file.")
            else: # handle errors
                print("\nPlease enter a valid integer")
        except ValueError:
            print("\nPlease enter a valid integer")

# output the docs to the user
def print_docs():
    print("\n", "="*10, "Docs", "="*10)
    for i, doc in enumerate(docs):
        print("[" + str(i) + "]", doc)

def choose_doc():
    global current_doc
    
    loop = True
    while loop:
        try: # loop while the user is choosing
            choice = int(input("\nEnter the index of the doc to use: "))

            if choice >= 0 and choice < len(docs):
                current_doc = choice
                loop = False
            else: # handle errors
                print("\nPlease enter a valid integer")
        except ValueError:
            print("\nPlease enter a valid integer")

# output the commands to the user
def print_commands(commands, title):
    print("\n", "="*10, title + " Commands", "="*10)
    for i, command in enumerate(commands):
        print("[" + str(i) + "]", command)

# display the selected input docs
def setup():
    choose_model(print_models())
    print_docs() # print available docs and allow
    choose_doc() # user to choose

# displays the general commands, as well as the selected mode
# commands
def display_tri_menu(commands, label):
    cls()
    print(docs[current_doc]) # print doc chosen at top
    print_commands(general_commands, "General")
    print_commands(commands, label) # print the commands

# handle a command from the general command list
def handle_general_command(command, menu_data):
    menu = True # ensure command is in command list
    if command in general_commands or command.lower() in general_commands:
        if command.lower() == general_commands[0].lower():
            print_docs()
            choose_doc()
            display_tri_menu(menu_data[0], menu_data[1])
            print("\nDoc changed successfully.")
        elif command.lower() == general_commands[1].lower():
            print_docs() # output the docs
        elif command.lower() == general_commands[2].lower():
            print("\n" + docs[current_doc]) # output the current doc
        elif command.lower() == general_commands[3].lower():
            f = open(path.join(get_output_data_dir(), "sample_swedish.txt"), "a", encoding="utf-8")
            for doc in docs: # export the docs into a file
                f.write(doc + "\n")
            f.write("\n")
            f.close() # export input data
            print("\nData successfully exported to", path.join(get_output_data_dir(), "sample_swedish.txt"))
        elif command.lower() == general_commands[4].lower():
            f = open(path.join(get_output_data_dir(), "sample_swedish.txt"), "a", encoding="utf-8")
            f.write(docs[current_doc] + "\n")
            f.write("\n")
            f.close() # export current doc
            print("\nData successfully exported to", path.join(get_output_data_dir(), "sample_swedish.txt"))
        elif command.lower() == general_commands[-2].lower():
            display_tri_menu(menu_data[0], menu_data[1]) # help
        elif command.lower() == general_commands[-1].lower():
            menu = False # end the menu loop
    else: # handle errors
        display_tri_menu(menu_data[0], menu_data[1])
        print("\nInvalid command entered.")

    return menu

def main_parser():
    display_tri_menu(parser_commands, "Parser")

    doc = nlp(docs[current_doc])
    
    loop = True
    while loop: # loop while user enters commands
        choice = input("\nEnter a command: ")

        if choice in parser_commands or choice.lower() in parser_commands:
            if choice.lower() == parser_commands[0].lower():
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_sents(doc.text) # print all sentences in doc
            elif choice.lower() == parser_commands[1].lower():
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_syntactic_info(doc, nlp) # print syntactic info
            elif choice.lower() == parser_commands[2].lower():
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_pos_frequency_list(doc) # print POS frequency
            elif choice.lower() == parser_commands[3].lower(): # list
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_word_frequency_list(doc, nlp)
            elif choice.lower() == parser_commands[4].lower():
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_tokens(doc, nlp) # print tokens in sentence
            elif choice.lower() == parser_commands[5].lower():
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_stopwords(doc) # print stopwords in sentence
            elif choice.lower() == parser_commands[6].lower():
                display_tri_menu(parser_commands, "Parser") # show menu
                sv_parser.print_dependency_skeleton(doc, nlp)
            elif choice.lower() == parser_commands[7].lower():
                display_tri_menu(parser_commands, "Parser")
                sv_parser.print_lemmatise_doc(doc, nlp) # print lemmas for tokens
        elif choice in general_commands: # end the menu
            loop = handle_general_command(choice, [parser_commands, "Parser"])
        else:
            display_tri_menu(parser_commands, "Parser")
            print("\nInvalid command entered.") # handle error

def main_ner():
    display_tri_menu(ner_commands, "NER")
    
    loop = True
    while loop: # loop while user enters commands
        choice = input("\nEnter a command: ")
        # if it is a valid command
        if choice in ner_commands or choice.lower() in ner_commands:
            if choice.lower() == ner_commands[0].lower():
                for tree in sv_ner.nltk_ne_trees(docs[current_doc], nlp):
                    print("\n", tree) # nltk entity trees
            elif choice.lower() == ner_commands[1].lower():
                current_sent = ""
                ne = sv_ner.spacy_ne(docs[current_doc], nlp)

                if len(ne) > 0: # if at least one named entity
                    for sent, ne in sv_ner.spacy_ne(docs[current_doc], nlp):
                        if current_sent != sent:
                            current_sent = sent # loop through sents and get NEs
                            print("\n" + str(current_sent))
                        print("")
                        print(ne) # spacy named entities
                else:
                    print("")
                    print("No entities found.")
            elif choice.lower() == ner_commands[2].lower():
                sv_ner.display_ne(docs[current_doc], nlp) # display NEs
        elif choice in general_commands:
            loop = handle_general_command(choice, [ner_commands, "NER"])
        else:
            display_tri_menu(ner_commands, "NER") # cls and display menus
            print("\nInvalid command entered.")

# menu for the information extraction module
def main_ie():
    display_tri_menu(ie_commands, "IE")
    
    sample_set = False
    loop = True
    while loop: # loop while user inputs commands
        choice = input("\nEnter a command: ").split()

        if len(choice) >= 2 and choice[1] == "--sample":
            sample_set = True # use a sample set for commands
        
        if choice[0] in ie_commands or choice[0].lower() in ie_commands:
            if choice[0] in ie_commands[0:len(ie_commands)]: # valid command to be handled
                display_tri_menu(ie_commands, "IE")
                if sample_set:
                    sample_set = False # use sample set

                    data = sample_training_set(10)
                    if len(data) > 0:
                        sv_ie.extract_info_sampleset(data, choice[0], nlp)
                    else:
                        print("A relevant corpus, such as SUC-3, is required to use this tool.")
                        exit()
                else: # use the current doc
                    sv_ie.extract_info(docs[current_doc], choice[0], nlp)
        elif choice[0] in general_commands:
            loop = handle_general_command(choice[0], [ie_commands, "IE"])
        else:
            display_tri_menu(ie_commands, "IE")
            print("\nInvalid command entered.")

# the main initial running of the program
def main():
    if len(argv) == 2 and argv[1] == "--help":
        help() # output help to user
    elif len(argv) == 3 and argv[2] != "" and argv[2].lower() != "--sample":
        _, extension = path.splitext(argv[2])
        # load user file contents
        if extension != "" and extension == ".txt":
            if path.exists(path.join(get_input_data_dir(), argv[2])):
                load_file(path.join(get_input_data_dir(), argv[2])) # load input file
            else:
                print("\nThe file", argv[2], "does not exist.")
                help() # handle incorrect file
        else:
            help()
    elif len(argv) == 3 and argv[2].lower() == "--sample":
        load_file("") # prompt sample data
    else:
        help()
    # loaded docs, so start programming
    if len(docs) > 0:
        cls()

        if unintentional_sample_data: # warn user their file was empty
            print("An empty file was provided, switching to sample data.\n")
        
        setup()
        # display certain menus depending on input mode
        if sys.argv[1] == "parse":
            main_parser()
        elif sys.argv[1] == "ner":
            main_ner()
        elif sys.argv[1] == "ie":
            main_ie()

def install():
    print("Checking for NLTK packages..")
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Installing missing package...")
        nltk.download("punkt")
        nltk.download("stopwords")

if __name__ == "__main__":
    install()
    main()
