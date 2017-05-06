# For features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

#For classificaition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB #notworking
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import Utils
from time import time
import logging
import joblib


def label_new_data(unlabeled_docs,original_docs):
      '''
        This method is supposed to used in production
        It requires that the model is trained and saved in a file locally. 
        It loads the model, predicts the labeles of unLabeled data, and stores the results
    '''
      # load the saved pipeline that includes both the vectorizer and the classifier and predict
      classifier = joblib.load('Saved_Shallow_Models\Model_ngrams_char_3_3.pkl')
      print("Classifier loaded . Predicting new samples ...")
      predicted_labels = classifier.predict(unlabeled_docs)
      res=[]

      print("Prepering data to store locally...")
      tmp_dataa=[]
      for docs in original_docs:
        #for i in range(len(docs)):
            tmp_dataa.append(docs)

      tmp_predy=[]
      for labels in predicted_labels:
        #for i in range(len(labels)):
            tmp_predy.append(labels)

      res=[]
      for i in range(len(original_docs)):
        stri = str(tmp_predy[i]) + "\t" + tmp_dataa[i]#["text"]
        res.append(stri) 
      print("Storing data locally ...")
      Utils.write_list_to_file("labeled_data.txt",res)
    
      return predicted_labels

 #Simple method to print the paramters
def print_settings(param):
    print("Settings : \n\n")
    print('#' * 80)
    method_description=""
    if(param["ngrams_bow"]["method"]=="ngrams"):
        method_description=param["ngrams_bow"]["feature_level"] + " level ngrams"
    else: 
        method_description=param["ngrams_bow"]["method"]
    print("Method: " + method_description)
    if(param["ngrams_bow"]["method"]=="ngrams"):
      print("ngram_range: ("+str(param["ngrams_bow"]["min_ngrams"])+ " ," + str(param["ngrams_bow"]["max_ngrams"]) + ")")
    print("Feature Value: " +param["ngrams_bow"]["feature_value"])
    print('#' * 80)
    print("\n")

def classify_fixed_train_test_set(train_data, train_labels, test_data,test_labels, feature_construction_method, parameters, show_progress=True):
    ''' 
        Document (text) Classification using the following features(fixed datasets)
        1) Bag-of-Words 
        2) N-grams 
    '''

    #Define Input parameters 
    max_num_features = parameters["ngrams_bow"]["max_num_features"]
    feature_level = parameters["ngrams_bow"]["feature_level"]
    min_ngram = parameters["ngrams_bow"]["min_ngrams"]
    max_ngram = parameters["ngrams_bow"]["max_ngrams"]
    print_settings(parameters)
    
    if(show_progress==True):
       logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Vectorizer is used to construct features. 
    if feature_construction_method =="bow":   # Bag-of-Words as features
       vectorizer = TfidfVectorizer(stop_words='english', non_negative=True,
                                         n_features=max_num_features)  # analyzer not accepted as paramter??? )

    elif feature_construction_method =="ngrams": # n-grams as features 
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                           n_features=max_num_features,
                                           analyzer=feature_level,
                                           ngram_range=(min_ngram,max_ngram))
       
    #elif feature_construction_method =="lsa": # mungon ,vectorizer
        #     #X_train_tfidf, X_test_tfidf = construct_tfidf_features(X_train, X_test)
        #     #X_train___,X_test___=construct_LSA_features(X_train_tfidf,X_test_tfidf)
        
    else:
        print("The method name is wrong ...")

    clf=LinearSVC()
    #clf = naive_bayes.GaussianNB()
    vec_clf = Pipeline([('vectorizer', vectorizer), ('pac', clf)])
    t0=time()
    print("Extracting features from the training data using a sparse vectorizer")
    vec_clf.fit(train_data,train_labels)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time() 
    y_pred = vec_clf.predict(test_data)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    print("Accuracy: "+ str(accuracy_score(test_labels,y_pred)))
    print("F1 : " + str(f1_score(test_labels, y_pred, average='macro')))

def lsa_classification(data,labels,number_of_components=100):
    svd = TruncatedSVD(number_of_components)
    lsa = make_pipeline(svd, Normalizer(copy=False))
 

    #STEP 2: 10-Fold Cross Validation of Data
    kf = KFold(n_splits=10,shuffle=True)
    train_times=[]
    test_times=[]
    accuracy=[]
    precision=[]
    recall = []  
    f1 = []
    clf=LinearSVC()
    #clf = KNeighborsClassifier()
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

    for train_index, test_index in kf.split(data):
        
        # Get the data for the current fold (10fold CV) 
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

           # Tfidf vectorizer:
    #   - Strips out 'stop words'
    #   - Filters out terms that occur in more than half of the docs (max_df=0.5)
    #   - Filters out terms that occur in only one document (min_df=2).
    #   - Selects the 10,000 most frequently occuring words in the corpus.
    #   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
    #     document length on the tf-idf values. 
   

        # Build the tfidf vectorizer from the training data ("fit"), and apply it to extract features from text data as well
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Run SVD on the training data, then project the training data.
        X_train_lsa = lsa.fit_transform(X_train_tfidf)

        # Run SVD on the training data, then project the training data.
        X_train_lsa = lsa.fit_transform(X_train_tfidf)

        #print("  done in %.3fsec" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


        # Now apply the transformations to the test data as well.
        X_test_lsa = lsa.transform(X_test_tfidf)

        t0=time()
        print("Extracting features from the training data using a sparse vectorizer")
        clf.fit(X_train_lsa,y_train)

        train_time = time() - t0
        train_times.append(train_time)
        print("train time: %0.3fs" % train_time)

        t0 = time() 
        y_pred = clf.predict(X_test_lsa)
        test_time = time() - t0
        test_times.append(test_time)
        print("test time:  %0.3fs" % test_time)

        accuracy.append(accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred, average="macro"))
        precision.append(precision_score(y_test, y_pred, average="macro"))
        recall.append(recall_score(y_test, y_pred, average="macro"))
        #print(classification_report(y_test, y_pred))

    #file_name = "Saved_Shallow_Models\Model_"+feature_construction_method+"_" + str(feature_level) + "_" + str(min_ngram)+"_" + str(max_ngram)+".pkl"
    #joblib.dump(vec_clf, file_name)

 
    print("Average Accuracy: " +  str(round(np.mean(accuracy)*100,2)))
    print("Average Precision: " + str(round(np.mean(precision)*100,3)))
    print("Average Recall: " +    str(round(np.mean(recall)*100,3)))
    print("Average F1 score: " + str(round(np.mean(f1)*100,3)))

    print("Train Time: " + str(round(np.sum(train_times),2)))
    print("Test Time: " + str(round(np.sum(test_times),2)))
    return np.mean(accuracy)



def run_document_classification(data, labels, parameters, show_progress=True):
    ''' 
        Document (text) Classification using the following features:
        1) Bag-of-Words 
        2) N-grams 
    '''

    #Define Input parameters 
    max_num_features = parameters["ngrams_bow"]["max_num_features"]
    feature_level = parameters["ngrams_bow"]["feature_level"]
    min_ngram = parameters["ngrams_bow"]["min_ngrams"]
    max_ngram = parameters["ngrams_bow"]["max_ngrams"]
    feature_value = parameters["ngrams_bow"]["feature_value"]
    feature_construction_method= parameters["ngrams_bow"]["method"]
    print_settings(parameters)
    
    if(show_progress==True):
       logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    #STEP 1: The data 
    data = np.array(data)
    labels = np.array(labels)

    
    # Vectorizer is used to construct features. 
    if feature_construction_method =="bow":   # Bag-of-Words as features
       if (feature_value=="frequency"):
           print("Using raw frequency as features ...") 
           vectorizer = CountVectorizer(max_features=max_num_features#stop_words='english', non_negative=True,
                                         )
       elif(feature_value=="TF-IDF"):
           vectorizer = TfidfVectorizer(#stop_words='english', non_negative=True,
                                             max_features=max_num_features)  # analyzer not accepted as paramter??? )

       elif (feature_value=="binary"):
            print("using binary features......>>>>>>")
            vectorizer = CountVectorizer(max_features=max_num_features, binary=True) 

    elif feature_construction_method =="ngrams": # n-grams as features 
        if (feature_value=="frequency"):
             print("Using raw frequency as features ...") 
             vectorizer = CountVectorizer(#stop_words='english', strip_accents ='ascii',#non_negative=True,
                                               max_features =max_num_features, #norm='l2',
                                               analyzer=feature_level,
                                               ngram_range=(min_ngram,max_ngram))
        elif(feature_value=="TF-IDF"):
            vectorizer = TfidfVectorizer(#stop_words='english', strip_accents ='ascii',#non_negative=True,
                                               max_features =max_num_features, #norm='l2',
                                               analyzer=feature_level,
                                               ngram_range=(min_ngram,max_ngram))
        elif (feature_value=="binary"):
            print("using binary features......>>>>>>")
            vectorizer = CountVectorizer( max_features =max_num_features, #norm='l2',
                                               analyzer=feature_level,
                                               ngram_range=(min_ngram,max_ngram), binary=True)

    elif feature_construction_method =="lsa": # mungon ,vectorizer
         return lsa_classification(data,labels,number_of_components=10)
        
    else:
        print("The method name is wrong ...")
    
    #STEP 2: 10-Fold Cross Validation of Data
    kf = KFold(n_splits=10,shuffle=True)
    train_times=[]
    test_times=[]
    accuracy=[]
    precision=[]
    recall = []  
    f1 = []
    clf=LinearSVC()
    #clf = KNeighborsClassifier()
    #clf=DecisionTreeClassifier()
    vec_clf = Pipeline([('vectorizer', vectorizer), ('pac', clf)])

    for train_index, test_index in kf.split(data):
        
        # Get the data for the current fold (10fold CV) 
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        t0=time()
        print("Extracting features from the training data using a sparse vectorizer")
        vec_clf.fit(X_train,y_train)

        train_time = time() - t0
        train_times.append(train_time)
        print("train time: %0.3fs" % train_time)

        t0 = time() 
        y_pred = vec_clf.predict(X_test)
        test_time = time() - t0
        test_times.append(test_time)
        print("test time:  %0.3fs" % test_time)

        accuracy.append(accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred, average="macro"))
        precision.append(precision_score(y_test, y_pred, average="macro"))
        recall.append(recall_score(y_test, y_pred, average="macro"))
        #print(classification_report(y_test, y_pred))

    file_name = "Saved_Shallow_Models\Model_"+feature_construction_method+"_" + str(feature_level) + "_" + str(min_ngram)+"_" + str(max_ngram)+".pkl"
    joblib.dump(vec_clf, file_name)

 
    avg_accuracy = round(np.mean(accuracy)*100,2)
    avg_precision = round(np.mean(precision)*100,2)
    avg_recall = round(np.mean(recall)*100,2)
    avg_f1 = round(np.mean(f1)*100,2)
    print("\nAverage Accuracy: " +  str(avg_accuracy))
    print("Average Precision: " + str(avg_precision))
    print("Average Recall: " +    str(avg_recall))
    print("Average F1 score: " +  str(avg_f1))

    print("Train Time: " + str(round(np.sum(train_times),2)))
    print("Test Time: " + str(round(np.sum(test_times),2)))

    return avg_accuracy,avg_precision,avg_recall,avg_f1


