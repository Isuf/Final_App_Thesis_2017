from __future__ import print_function
from time import time
import Utils as Utils
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.model_selection import train_test_split


import pickle 
import Configuration
import matplotlib.pyplot as plt

def print_latex_topics(model, feature_names, n_top_words=10,n_topics=10):
        """Prints the n top words from the model in a Latex friendly format.

        :param model: The model
        :param feature_names: The feature names
        :param n_words: Number of top words
        """
        #if n_words is not None:
        #    self.n_top_words = n_words

        top_words = []
        for topic in model.components_:
            top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

        delimiter = "&"
        new_line ="\\\\"
        hline = "\hline"
        header = ""
        for i in range(n_topics):
            header += "Topic #{}".format(i)
            if i != n_topics-1:
                header += " {} ".format(delimiter)
        header += "{} {}".format(new_line, hline)

        string_top_words = []
        for n in range(10):
            buf = ""
            for i in range(n_top_words):
                buf += "{}".format(top_words[i][n], delimiter)
                if i != n_topics-1:
                    buf += " {} ".format(delimiter)

            buf += new_line
            string_top_words.append(buf)
        print(header)
        for line in string_top_words:
            print(line)

        header = ""
        for i in range(0, 10):
            header += "Topic {}".format(i)
            if i != len(top_words) - 1:
                header += " {} ".format(delimiter)
        header += "{} {}".format(new_line, hline)

        string_top_words = []

        for n in range(10):
            buf = ""
            for i in range(0, 10):
                buf += "{}".format(top_words[i][n], delimiter)
                if i != 9:
                    buf += " {} ".format(delimiter)

            buf += new_line
            string_top_words.append(buf)

 

def print_top_words(model, feature_names, n_top_words):
    '''Prints the n top words from the model.

    :param model: The model
    :param feature_names: The feature names
    :param n_top_words: Number of top words
    '''
    store_topics=[]
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_keywords=" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(topic_keywords)
        store_topics.append(topic_keywords)
    print()
    Utils.write_list_to_file("topic_keywords.txt",store_topics)

def get_perplexity(held_out_data, lda,max_features):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', analyzer='word',max_features=max_features)  # , max_features=30)
    data = vectorizer.fit_transform(held_out_data)
    return lda.perplexity(data)


def run_LDA(corpus,num_topics=10,maximum_iterations=10, learn_offfset=10.0,max_features=100000000,max_df=0.95,min_df=2):
      
    t0 = time()

    print("Extracting tf features for LDA...")
    #tf_vectorizer = CountVectorizer(max_df=max_df, 
    #                                min_df=min_df, 
    #                                stop_words='english', 
    #                                analyzer='word',max_features=max_features)
    tf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features,max_df=max_df,min_df=min_df) 

    print("Fitting LDA models with tf features")
    tf = tf_vectorizer.fit_transform(corpus)
    print("     done in %0.3fs." % (time() - t0))

    #2. Define the model
    lda = LatentDirichletAllocation(n_topics=num_topics, 
                                    learning_method='online',
                                    max_iter=maximum_iterations,
                                    learning_offset=learn_offfset,
                                    random_state=0)

    # 3. Train the model
    t0 = time()
    print("training the model")
    lda.fit(tf)
    print("   done in %0.3fs." % (time() - t0))

    print("perplexity:" + str(lda.perplexity(tf,lda.transform(tf))))
     # Print LDA model
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()

    print_top_words(lda, tf_feature_names, n_top_words=11)

    return lda, tf,tf_vectorizer


def get_topdocuments_per_topic(lda,tf,num_topics, number_of_documents,topic_number):
    #print sample documents per topic
    

    doc_topic= lda.transform(tf)
    num_docs, num_topics= doc_topic.shape
    dic={}
    for i in range(num_topics):
        dic[str(i)] =[ doc_topic[doc_idx][i]   for doc_idx in range(num_docs)]

    ind =np.asarray(dic[str(topic_number)]).argsort()[-number_of_documents:][::-1]

    print("########################3\n\n")
    for i in range(len(ind)):
        print(data[ind[i]])
        print()


if __name__ == "__main__":

    params = Configuration.Parameters()
    file_locations=params.file_locations
    param = params.param
  
    clean_string = True
    remove_stop_words=False
    remove_numbersonly_words=False
    retrain=True
    evaluate_perplexity=False

    data_source="Security_Only" #Security_Only  All_Binary   #Not_Security
    if data_source=="Security_Only":
       data_folder_security=["results\\Security_data.txt"]
       dataset="Binary_SecurityOnly_trigrams"
    elif data_source=="All_Binary": 
       data_folder_security=["results\\Security_data.txt","results\\irrelevant_data.txt"]
       dataset="Binary_All_trigrams"
    elif data_source=="Not_Security":
        data_folder_security=["results\\irrelevant_data.txt"]
        dataset="Binary_NotSecurity_trigrams"
   
    n_topics=10
    max_features=1000
    max_df=0.85
    min_df=5
    max_iterations = 10
 
    print("###############################################################################################")
    print("Dataset: " + dataset)
    print("Number of Topics: " + str(n_topics))
    print("Max Number of features : " + str(max_features))
    print("MAX-DF: "+ str(max_df))
    print("MIN_DF:" +str(min_df))
    print("MaX_Itarations: "  + str(max_iterations))
    print("###############################################################################################")

    docs, vocab,_ =  Utils.load_data_LDA(data_folder_security,clean_string=clean_string,remove_numbers=remove_numbersonly_words,split_for_cv=True)
    max_l= Utils.get_dataset_details(docs,vocab)
    data = [ sent["text"]  for sent in docs ]

    model_name=dataset+"_"+str(n_topics)+"topics_"+str(max_features)+"features.pkl"
    if retrain==False: # Load a model that already exist. Make sure to use the same data as used when model was built
        x=Utils.load_pickle("lda_models\\binary\\"+model_name)
        lda, tf,tf_vectorizer=x[0],x[1],x[2]
        print_top_words(lda,tf_vectorizer.get_feature_names(),10)
        print_latex_topics(lda,tf_vectorizer.get_feature_names(),10)
    else : 
        
         if evaluate_perplexity:
           ##docs, vocab,_ = Utils.load_data_LDA(data_folder_security,clean_string=clean_string,remove_numbers=remove_numbersonly_words,split_for_cv=True)
           ##data = [ sent["text"]  for sent in docs  if sent["split"]>0]
           ##held_out_data = [sent["text"]  for sent in docs  if sent["split"]==0]
           ##Utils.write_list_to_file("SecurityOnly_Data.txt",data)
           ##Utils.write_list_to_file("SecurityOnly_HeldoutData.txt",held_out_data)
           data_folder=["SecurityOnly_Data.txt"]
           held_out_folder=["SecurityOnly_HeldoutData.txt"]
           docs, vocab,_ = Utils.load(data_folder,clean_string=clean_string,remove_stop_words=remove_stop_words,remove_numbers=remove_numbersonly_words, 
                                   split_for_cv=True)
           held_out_docs, vocab,_ = Utils.load(held_out_folder,clean_string=clean_string,remove_stop_words=remove_stop_words,remove_numbers=remove_numbersonly_words
                                    ,split_for_cv=True)
           data = [ sent["text"]  for sent in docs ]

         lda, tf,tf_vectorizer=run_LDA(data,num_topics=n_topics,maximum_iterations=max_iterations,max_features=max_features,max_df=max_df,min_df=min_df)
         #Utils.save_pickle([lda, tf,tf_vectorizer],"LDA_Models_BinaryDataset\\"+model_name)
         
         if evaluate_perplexity:
           held_out_data = [ sent["text"]  for sent in held_out_docs ]
           print("\n\nPerplexity on held-out documents:    " + str(get_perplexity(held_out_data,lda,max_features)))

    
         get_topdocuments_per_topic(lda,tf,n_topics,20,9)
  