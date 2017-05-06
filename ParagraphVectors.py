from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import multiprocessing
cores = multiprocessing.cpu_count()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import statsmodels.api as sm
import random 
import pickle 
import numpy as np 
import time 

def create_vectors_from_models (model_DM,model_DBOW,data):
    num_examples= len(data)
    vector_dimensionality= len(model_DM.docvecs[0])
    data_vectors = np.zeros((num_examples,vector_dimensionality))
    for i in range(num_examples):
        doc_indx = str(i)
        data_vectors[i] = model_DM.docvecs[int(doc_indx)] + model_DBOW.docvecs[int(doc_indx)]

    return np.asarray(data_vectors)

def run_paragraph_vector_classification(data,labels):
    ###################### Training Models  ###################
    data = [TaggedDocument(list(data[i]),[i]) for i in range(0,len(data))]
    Doc2VecTrainID = list(range(0,len(data)))
    random.shuffle(Doc2VecTrainID)
    trainDoc = [data[id] for id in Doc2VecTrainID]
    
    model_DM = Doc2Vec(size=400, window=10, min_count=1, sample=1e-4, negative=5, workers=cores,  dm=1, dm_concat=1 )
    model_DBOW = Doc2Vec(size=400, window=10, min_count=1, sample=1e-4, negative=5, workers=cores, dm=0)
    model_DM.build_vocab(trainDoc)
    model_DBOW.build_vocab(trainDoc)
    print("training ...")
    for it in range(0,25):
        print("epoch : " + str(it) )
        random.shuffle(Doc2VecTrainID)
        trainDoc = [data[id] for id in Doc2VecTrainID]
        model_DM.train(trainDoc)
        model_DBOW.train(trainDoc)

    #Save the models for later use 
    #pickle.dump([model_DM,model_DBOW ], open("paragraphvect_models.pkl", "wb"),protocol=4)

    #model_DM,model_DBOW=pickle.load(open("paragraphvect_models.pkl","rb"))

    ###################### CLASSIFICAATION USING LOGISTIC REGRESSiON ###################33
    print("classifciation part .... ")

     #STEP 2: 10-Fold Cross Validation of Data
    kf = KFold(n_splits=10,shuffle=True)

    accuracy=[]
    precision=[]
    recall = []  
    f1 = []

    data_vectors= create_vectors_from_models(model_DM,model_DBOW,data)
    labels = np.asarray(labels)
    for train_index, test_index in kf.split(data_vectors):
            # Get the data for the current fold (10fold CV) 
            X_train, X_test = data_vectors[train_index], data_vectors[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            t0=time.time()
            train_targets, train_regressors =y_train, X_train
            train_regressors = sm.add_constant(train_regressors)
            predictor = LogisticRegression(multi_class='multinomial',solver='lbfgs')
            predictor.fit(train_regressors,train_targets)

            test_regressors = X_test
            test_regressors = sm.add_constant(test_regressors)
            y_pred = predictor.predict(test_regressors)

            test_time = time.time() - t0
            print("test time:  %0.3fs" % test_time)

            accuracy.append(accuracy_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred, average="macro"))
            precision.append(precision_score(y_test, y_pred, average="macro"))
            recall.append(recall_score(y_test, y_pred, average="macro"))


  
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
