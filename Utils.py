
import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import pickle 
import gensim
import os

from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

#########################################################################################################
######                This file contains methods used to load/store the data.                     #######
######                Author : Isuf Deliu                                                         #######
######                         Norway, 2017                                                       #######
#########################################################################################################





def load(data_folder, clean_string=False, remove_stop_words=False, split_for_cv=False):
    '''
        Load all the files from data_folder  in a format like 
        { "text" :___, "y": ___, "num_words":____, "split":____}
    '''
    docs = []
    original_docs=[]
    vocab = defaultdict(float)
    num_zero_sentences=0
    for i in range(len(data_folder)):
        
        with open(data_folder[i], "r",encoding="utf-8",errors="ignore") as f:
             for line in f:
                 doc = []
                 doc.append(line.strip())
                 original_docs.append(line.strip())
                 text = " ".join(doc)#.lower() #remove all whitespace + lowercase
                 if clean_string:
                    text = clean_str(" ".join(doc))

                 if remove_stop_words: 
                    text = ' '.join([ '' if word.isdigit() else word  for word in text.split()[0:250] if word not in cachedStopWords])  
                 else:
                    text = ' '.join([ '' if word.isdigit() else  word  for word in text.split()[0:250]])#if len(word) >1])

                 #if remove_stop_words: 
                 #   text = ' '.join([ '' if word.isdigit() else word  for word in text.split() if word not in cachedStopWords])  
                 #else:
                 #   text = ' '.join([ '' if word.isdigit() else  word  for word in text.split()])#if len(word) >1])

                 #print("\n" + remove_all_whitespaces(text))
                 if(len(text.split())>-1):
                     words = set(text.split())
                     for word in words:
                        vocab[word] += 1
                     if split_for_cv:
                         datum ={ "y": i,  "text" : text,   "num_words": len(text.split()),   "split": np.random.randint(0,10) , "num_chars": len(text)}
                     else : 
                         datum ={ "y": i,  "text" : text,   "num_words": len(text.split()) }
                     docs.append(datum)
                    
    print("Number of empty sentences : " + str(num_zero_sentences))
    return docs,vocab,original_docs

def remove_all_whitespaces(text):
    return  " ".join(text.split())

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets 
    Every dataset is lower 
     Source : https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py 

    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)    
    #print(string)

    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 

    #Replace two or more whitespace char (\t\n\r\f\v) with a single space
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_dataset_details(docs,vocab={}):
    max_l = np.max(pd.DataFrame(docs)["num_words"])
    avg_l = np.mean(pd.DataFrame(docs)["num_words"])
    min_l = np.min(pd.DataFrame(docs)["num_words"])
    max_char = np.max(pd.DataFrame(docs)["num_chars"])
    avg_char= np.mean(pd.DataFrame(docs)["num_chars"])
    min_char = np.min(pd.DataFrame(docs)["num_chars"])
    print("data loaded!")
    print("         number of documents: " + str(len(docs)))
    print("         vocab size: " + str(len(vocab)))
    print("         max document length: " + str(max_l))
    print("         avg document length: " + str(avg_l))
    print("         min document length: " + str(min_l))

    print("         max char length: " + str(max_char))
    print("         avg char length: " + str(avg_char))
    print("         min char length: " + str(min_char))
    return max_l
#####################################
def loadfile(fileName): 
    '''
        Loads a file and saves its content as a list  [ "this is test 1", " this is test 2"]
    '''
    with open(fileName,encoding="utf8",errors='ignore') as f:
         content = f.readlines()
         content = [x for x in content] 
    return content

def write_list_to_file(fileName, list, mode="w"):    
    ''' 
    Write a list to a local file 
    ''' 
    with open(fileName,mode, encoding="utf-8") as f:
         for item in list:
             f.write(item +'\n') #item+'\n'


def write_to_file(fileName, text, mode="w"):
    ''' Write the "text" to a local file '''
    with open(fileName,mode, encoding='utf-8') as f:
         f.write(text) # python will convert \n to os.linesep 

def save_pickle(self, path):
    with open(path, 'wb') as f:
        pickle.dump(self, f)
    #logger.info('save model to path %s' % path)
    return None

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_data_by_class_binary(data_file):
    security_data=[]
    irrelevant_data=[]
    with open(data_file,encoding="utf8",errors='ignore') as f:
         content = f.readlines()
         for doc in content:
             if(doc[0] =="0"):
               irrelevant_data.append(doc[2:])
             else: 
              security_data.append(doc[2:])
    store("Results\security_data.txt",security_data)
    store("Results\irrelevant_data.txt",irrelevant_data)
def split_data_by_class_mc (data_file):
    irrelevant_data=[]
    zero_Credentials=[]
    one_DDOS =[]
    two_Keyloggers=[]
    three_Crypters=[]
    four_Trojans=[]
    five_SQLi=[]
    six_SPAM =[]
    seven_RAT=[]
    with open(data_file,encoding="utf8",errors='ignore') as f:
         content = f.readlines()
         for doc in content:
             if(doc[0] =="8"):
               irrelevant_data.append(doc[2:])
             elif(doc[0] =="0"):
              zero_Credentials.append(doc[2:])
             elif(doc[0] =="1"):
              one_DDOS.append(doc[2:])  
             elif(doc[0] =="2"):
              two_Keyloggers.append(doc[2:])   
             elif(doc[0] =="3"):
              three_Crypters.append(doc[2:])
             elif(doc[0] =="4"):
              four_Trojans.append(doc[2:]) 
             elif(doc[0] =="5"):
              five_SQLi.append(doc[2:]) 
             elif(doc[0] =="6"):
              six_SPAM.append(doc[2:]) 
             elif(doc[0] =="7"):
              seven_RAT.append(doc[2:]) 


    store("Results\\A_Credentials.txt",zero_Credentials)
    store("Results\\B_DDOS.txt",one_DDOS)
    store("Results\\C_Keyloggers.txt",two_Keyloggers)
    store("Results\\D_Crypters.txt",three_Crypters)
    store("Results\\E_Trojans.txt",four_Trojans)
    store("Results\\F_SQLi.txt",five_SQLi)
    store("Results\\G_SPAM.txt",six_SPAM)
    store("Results\\H_RAT.txt",seven_RAT)
    store("Results\\I_irrelevant_data.txt",irrelevant_data)
def store(file,data):
     with open(file,"w",encoding="utf-8") as f:
         for item in data:
             f.write(item )

def python_to_ipython():

    """Create a notebook containing code from a script.
    Run as:  python make_nb.py my_script.py
    """
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell

    python_fileName = "visualize_topics.py"
    ipython_fileName ="topic_models.ipynb"

    nb = new_notebook()
    with open(python_fileName) as f:
        code = f.read()

    nb.cells.append(new_code_cell(code))
    nbformat.write(nb, ipython_fileName)
    print("IPython Notebook created Successfully") 

#split_data_by_class_mc("labeled_data.txt")
#split_data_by_class_binary("labeled_data.txt")
#python_to_ipython()

