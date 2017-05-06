import numpy as np
import pandas as pd
import Utils
import Configuration 

import Distributional_Representation
import TopicModeling as LDA
import ParagraphVectors as PV 
import time

'''
    CNN  : for Convelutional Neural Networks 
    'distribuional' for others 
'''
classification_method="distribuional"  #distribuional #CNN #LDA #doc2vec

params = Configuration.Parameters()
file_locations=params.file_locations
param = params.param

clean_string = True
remove_stop_words=True
testMode=True 

import glob
data_folder=glob.glob("D:\\Tema NTNU\\Data\\Delivery\\Opinosis\\topics\\*.data")

testing_data_folder=["Results\\security_data.txt"]#params.baseline_locations["Nulled_Test"] #
data_folder=params.baseline_locations["Nulled_MC"]

if(classification_method=="distribuional"):
    '''
        bow : bag-of-words 
        ngrams : n-grams (word or character level; Check config file) 
        lsa : for lsa classification
    '''

    if(testMode):
        print("Loading unLabeled Data...")
        test_docs, vocab,original_docs=Utils.load(testing_data_folder,clean_string=clean_string,remove_stop_words=remove_stop_words,split_for_cv=True)
        max_l= Utils.get_dataset_details(test_docs,vocab)
        test_docs = [ sent["text"] for sent in test_docs]
        Distributional_Representation.label_new_data(test_docs,original_docs)

    else:
        docs, vocab,_ = Utils.load(data_folder,clean_string=clean_string,remove_stop_words=remove_stop_words,split_for_cv=True)
        max_l= Utils.get_dataset_details(docs,vocab)
        labels = [sent["y"] for sent in docs]
        data = [ sent["text"] for sent in docs]
        tmp =[str(sent["num_words"]) for sent in docs]

        ##Utils.write_list_to_file("Num_words.txt",tmp)
        ##print(" STORED>>>>>")
    
        accs=[]
        recalls=[]
        precisions=[]
        f1s=[]
        for i in range(10):
            avg_accuracy,avg_precision,avg_recall,avg_f1= Distributional_Representation.run_document_classification(data, labels,param,True)
            accs.append(avg_accuracy)
            recalls.append(avg_recall)
            precisions.append(avg_precision)
            f1s.append(avg_f1)
        # Distributional_Representation.classify_fixed_train_test_set(X_train, y_train,X_test,y_test,feature_construction_method, param, True)
       
        print("########################################################################################################" )
        print(" \n\nFinal Results after 10 iterations : " )
        print("Avg Accuracy: " + str(round(np.mean(accs),2)))
        print("Avg Prescion: " + str(round(np.mean(precisions),2)))
        print("Avg Recall: " + str(round(np.mean(recalls),2)))
        print("Avg F1: " + str(round(np.mean(f1s),2)))

elif classification_method=="LDA":
      data_folder_security=[params.baseline_locations["tmp"][0]]# ["results\Security_data.txt"] #[params.baseline_locations["Nulled_Binary"][1]]
      docs, vocab,_= Utils.load(data_folder_security,clean_string=clean_string,remove_stop_words=remove_stop_words,split_for_cv=True)
      max_l= Utils.get_dataset_details(docs,vocab)
      labels = [sent["y"] for sent in docs]
      data = [ sent["text"] for sent in docs]

      t = time.time()
      LDA.run_LDA(data,evaluate=False,evaluation_data=[],
                  num_topics=30, n_top_words=10, max_n_features=50000, 
                  max_df=0.75, min_df=1, max_iter=5,
                  show_topics=True, store=False)
      print("     done in %0.3fs." % (time.time()- t))
elif classification_method=="doc2vec": 

      data_folder=params.baseline_locations["Nulled_Binary"]
      docs, vocab,_ = Utils.load(data_folder,clean_string=True,remove_stop_words=False,split_for_cv=True)
      max_l= Utils.get_dataset_details(docs,vocab)
      labels = [sent["y"] for sent in docs]
      data = [ sent["text"].split() for sent in docs]
      PV.run_paragraph_vector_classification(data, labels)