from __future__ import print_function
from time import time
import Utils as Utils
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.model_selection import train_test_split

#import pyLDAvis
#import pyLDAvis.sklearn 
#pyLDAvis.enable_notebook()
import pickle 


n_samples = 10000
def print_topics (lda_model, tf_vectorizer, num_words_per_topic, store_topics=False):
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_no ="Topic #%d:" % topic_idx
        topic_kw = " ".join([tf_feature_names[i] for i in topic.argsort()[:-num_words_per_topic - 1:-1]])
        print(topic_no)
        print(''.join([tf_feature_names[i] + ' ' + str(round(topic[i]/ sum(topic),4 ))  +' | ' for i in topic.argsort()[:-num_words_per_topic - 1:-1]]))
        #print(topic_kw)
        print()
        if(store_topics==True):
          Utils.write_to_file("topics.txt",topic_no+"\n","a")
          Utils.write_to_file("topics.txt",topic_kw+"\n","a")

def print_lda_parameters(num_topics=10, num_top_words=10, num_features=30000,heldout_data_size =0.33):
    print("Running Topic Models...")
    print("          Number of Topics : " + str(num_topics))
    print("          Number of Words per Topics : " + str(num_top_words))
    print("          Maximum  Number of Features : " + str(num_features))

def store_topics(lda,tf,corpus_docs):
     ##Store the data according to their topiccs
    print("Storing data to files ")
    doc_topic= lda.transform(tf)
    num_docs, num_topics= doc_topic.shape
    for doc_idx in range(num_docs):
        doc_most_pr_topic = np.argmax(doc_topic[doc_idx])
        document_text = corpus_docs[doc_idx] 
        file_location = "Results/topic_" + str(doc_most_pr_topic) + "_documents.txt"
        Utils.write_to_file(file_location,document_text,"a")
    print("data stored to corrensponding files") 

#def visualize_topics ():
#    x = pickle.load(open("lda_model_Nulled.pkl","rb"))
#    print("LDA model loaded ")
#    lda, tf, tf_vectorizer= x[0], x[1], x[2]
#    lda.fit(tf)
#    pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
#    print("prepared...!")

############ Main method #########
def build_LDA_model(corpus_data, num_topics=10, max_n_features =30000, max_df=0.95, min_df=1, max_iter=5):
    
    ''' 
         common English words are removed,
         words occurring in min_df(e.g=1) documents are removed 
         words occurring in at least 95% of the documents are removed.

    '''
   
    t0 = time()
    #1  Feature Extaction from raw data. Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,
                                    max_features=max_n_features,
                                    stop_words='english')
    print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (len(corpus_data), max_n_features))
    tf = tf_vectorizer.fit_transform(corpus_data)
    print("     done in %0.3fs." % (time() - t0))


    #2. Build the model
    lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=max_iter,
                                    learning_method='online',
                                    #learning_offset=50.,
                                    random_state=0)
    

    # 3. Train the model
    t0 = time()
    print("training the model")
    lda.fit(tf)
    print("   done in %0.3fs." % (time() - t0))

    return lda, tf,tf_vectorizer


def run_LDA(corpus_data, evaluate=False,evaluation_data=[],  num_topics=10, n_top_words=10, max_n_features =30000, max_df=0.95, min_df=1, max_iter=5,
            show_topics=True,store=True):
    print_lda_parameters(num_topics, n_top_words, max_n_features)

    # 1. Preprocessing the data 

    # 2. Build the Model 
    lda,tf,tf_vectorizer=build_LDA_model(corpus_data,
                                         num_topics, max_n_features,
                                         max_df, min_df,max_iter)
    
    # 3. Show Topics 
    if (show_topics==True):
       print_topics(lda,tf_vectorizer,n_top_words, store_topics)

    # 4. Evaluate Topics 
    if(evaluate==True):
       if len(evaluation_data)==0:
          print("Eorr: You must specify evaluation data")
       else:
          perplexity=held_out_perplexity(lda,tf_vectorizer, evaluation_data)
          print("Perplixity : " + str(perplexity))
     #pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

    if(store==True):
       store_topics(lda, tf, corpus_data)

    pickle.dump([lda,tf,tf_vectorizer ], open("lda_model_Nulled.pkl", "wb"))
    return lda,tf,tf_vectorizer 
################################################

#visualize_topics()