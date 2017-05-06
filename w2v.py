from __future__ import print_function
#from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import gensim
import multiprocessing
#https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/data_helpers.py

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
def get_random_vectors(word_vectors_model,vocabulary,vector_size=300):
    word_idx_map = dict()
    i=1
    wordInModel=0
    embedding_weights=[]
    embedding_weights = np.zeros(shape=(len(vocabulary)+1, vector_size), dtype='float32')            
    embedding_weights[0] = np.zeros(vector_size, dtype='float32')
    word_idx_map = dict()
    for word in vocabulary: 
        if word not in word_vectors_model:
           rand_vector = np.random.uniform(-0.25, 0.25, vector_size)
           embedding_weights[i]=rand_vector
        else:
            wordInModel +=1
            embedding_weights[i] = word_vectors_model[word]
        word_idx_map[word] = i
        i += 1
    return embedding_weights

def test_load_TrainedOnDataVectors(source="ToData"):
    model_name="D:\Tema NTNU\Data\Vector_Models\TOData_word2vec_300.bin"
    if(source=="Google"):
        model_name="D:\Tema NTNU\Data\Vector_Models\Google_word2vec_300.bin"
    elif source=="Glove":
         model_name="D:\Tema NTNU\Data\Vector_Models\Glove_word2vec_300.bin"
    model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)  

    word=input('\nEnter a word: ')
    print(" Similar to "+ word) 
    print(model.most_similar(word))

    while(input("Another word(y-for yes)" =="y")):
        word=input('\nEnter a word: ')
        print(" Similar to "+ word) 
        print(model.most_similar(word))


def build_w2v_model(data, vectors_source="Google",binary_format=True, vector_size=300, min_word_count=1, context=5):

    vectors_dir = 'D:\Tema NTNU\Data\Vector_Models'
    model_name="{:s}_word2vec_{:d}".format(vectors_source,vector_size)
    model_name = join(vectors_dir, model_name+ (".bin" if binary_format else ".txt"))

    print("Model Name: " + model_name)
    if exists(model_name):
        print(" Vectors already exists")
        model =gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=binary_format)   
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        print(" Building word2vec model from data ..." )
        sentences= [sent["text"].split() for sent in data]
        print(sentences[0])
        model =gensim.models.word2vec.Word2Vec(sentences, 
                                                size=vector_size,
                                                min_count=min_word_count,
                                                window=context, 
                                                workers=multiprocessing.cpu_count(),
                                                sg = 0, #Training Algorithm 1-Skip Gram 0 - CBOW)
                                                hs = 0,  # Use of Hierarchical Sampling Default 1 
                                                negative = 10   # Use of negative sampling; Default 0 (usually between 5-20)
                                                )
        model.init_sims(replace=True) #for memory efficiency

        # Saves the model for later use. You can load it later using Word2Vec.load()
        if not exists(vectors_dir):
            os.mkdir(vectors_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        model.wv.save_word2vec_format(model_name,binary=binary_format)

    return model

def get_embedding_weights(data,vocabulary, vectors_source="Google",binary_format=True, vector_size=300, min_word_count=1, context=5):
     
    #Get the word2vec model
    model = build_w2v_model(data, vectors_source,binary_format=binary_format, vector_size=vector_size, min_word_count=1, context=5)
    
    embedding_weights = np.zeros(shape=(len(vocabulary)+1, vector_size), dtype='float32') 
    embedding_weights[0] = np.zeros(vector_size, dtype='float32')
    word_idx_map = dict()
    i=1
    wordInModel=0
    for word in vocabulary: 
        # The vectors of the words not in the model are assigned at random
        if word not in model:
           rand_vector = np.random.uniform(-0.25, 0.25, vector_size)
           embedding_weights[i]=rand_vector
        else:
            wordInModel +=1
            embedding_weights[i] = model[word]
        word_idx_map[word] = i
        i += 1
    print("Number of words already in word2vec model :" + str(wordInModel))
    return embedding_weights,word_idx_map



#test_load_TrainedOnDataVectors("ToData")