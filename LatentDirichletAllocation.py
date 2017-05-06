from __future__ import print_function
from time import time
import Utils as Utils
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.model_selection import train_test_split


import pickle 
import Configuration


def print_top_words(model, feature_names, n_top_words):
    '''Prints the n top words from the model.

    :param model: The model
    :param feature_names: The feature names
    :param n_top_words: Number of top words
    '''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def run_LDA(corpus,num_topics=10,maximum_iterations=10, learn_offfset=10.0):
      
    t0 = time()

    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, 
                                    min_df=2, 
                                    stop_words='english', 
                                    analyzer='word')

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

     # Print LDA model
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()

    print_top_words(lda, tf_feature_names, n_top_words=10)

    return lda, tf,tf_vectorizer

if __name__ == "__main__":

    params = Configuration.Parameters()
    file_locations=params.file_locations
    param = params.param
  
    clean_string = True
    remove_stop_words=False
    retrain=True

    dataset="Security_data"
    n_topics=10

    data_folder_security=["results\C_Keyloggers.txt"] #[params.baseline_locations[dataset][0]]
    #[params.baseline_locations["tmp"][0]]# ["results\Security_data.txt"] #[params.baseline_locations["Nulled_Binary"][1]]
    docs, vocab,_ = Utils.load(data_folder_security,clean_string=clean_string,remove_stop_words=remove_stop_words,split_for_cv=True)
    max_l= Utils.get_dataset_details(docs,vocab)
    data = [ sent["text"] for sent in docs]
    
    if(retrain):
        lda, tf,tf_vectorizer=run_LDA(data,num_topics=n_topics,maximum_iterations=10)
        Utils.save_pickle([lda, tf,tf_vectorizer],dataset+str(n_topics)+"_topics.pkl")
    else:
        x=Utils.load_pickle("lda_models\\"+dataset+str(n_topics)+"_topics.pkl")
        lda, tf,tf_vectorizer=x[0],x[1],x[2]

    doc_topic= lda.transform(tf)
    num_docs, num_topics= doc_topic.shape
    print(doc_topic[1])
    dic={}
    for i in range(num_topics):
        dic[str(i)] =[ doc_topic[doc_idx][i]   for doc_idx in range(num_docs)]

    ind =np.asarray(dic["3"]).argsort()[-5:][::-1]
    print(ind)

    print("########################3\n\n")
    for i in range(len(ind)):
        print(data[ind[i]])
        print()
    #for doc_idx in range(num_docs):
    #    doc_most_pr_topic = np.argmax(doc_topic[doc_idx])
    #    document_text = corpus_docs[doc_idx] 
    #    file_location = "Results/topic_" + str(doc_most_pr_topic) + "_documents.txt"
    #    Utils.write_to_file(file_location,document_text,"a")
    #print("data stored to corrensponding files") 
