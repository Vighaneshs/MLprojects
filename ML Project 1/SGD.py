#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import os
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt_tab')

# enron1 is 1st, enron2 is 2nd and enron4 is 3rd
no_of_datasets = 3


# In[2]:


#To read data from the folders
train_data_folders = ["\\enron1_train\\enron1\\train", "\\enron2_train\\train", "\\enron4_train\\enron4\\train"]
test_data_folders = ["\\enron1_test\\enron1\\test", "\\enron2_test\\test", "\\enron4_test\\enron4\\test"]

train_data_folders=[ os.getcwd() + t for t in train_data_folders]
test_data_folders=[ os.getcwd() + t for t in test_data_folders]


#0, 1 ,2 indices of list will be for the 3 datasets
#Each list is vocabulary for i+1th dataset 
train_vocab_lists = [[] for i in range(no_of_datasets)]
#counter to help creating vector of words later
no_message = []
for dataset, train_folder in enumerate(train_data_folders):
    #-1 is for Subject which I am removing
    count = 0
    for filename in os.listdir(train_folder + "\\ham"):
        count += 1
        with open(os.path.join(train_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    if word not in train_vocab_lists[dataset]:
                        train_vocab_lists[dataset].append(word)
                
                    
    for filename in os.listdir(train_folder + "\\spam"):
        count += 1
        with open(os.path.join(train_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    if word not in train_vocab_lists[dataset]:
                        train_vocab_lists[dataset].append(word)
    no_message.append(count)


# In[3]:


bow_matrices = []

test_bow_matrices = []

#iterating over number of messages to create bag of word matrices for all 3 datasets
#Last column of the numpy array is the inference i.e 0 for ham and 1 for spam
for i in range(no_of_datasets):
    bow_matrices.append(np.zeros((no_message[i], len(train_vocab_lists[i])+1)))

for i in range(no_of_datasets):
    test_bow_matrices.append(np.zeros((no_message[i], len(train_vocab_lists[i])+1)))


for dataset, train_folder in enumerate(train_data_folders):
    msg_index = 0
    for filename in os.listdir(train_folder + "\\ham"): 
        with open(os.path.join(train_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    (bow_matrices[dataset])[msg_index][train_vocab_lists[dataset].index(word)] += 1
                    (bow_matrices[dataset])[msg_index][-1] = 0
            msg_index += 1
                
                    
    for filename in os.listdir(train_folder + "\\spam"):
        with open(os.path.join(train_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    (bow_matrices[dataset])[msg_index][train_vocab_lists[dataset].index(word)] += 1
                    (bow_matrices[dataset])[msg_index][-1] = 1
            msg_index += 1

for dataset, test_folder in enumerate(test_data_folders):
    msg_index = 0
    for filename in os.listdir(test_folder + "\\ham"): 
        with open(os.path.join(test_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    try:
                        (test_bow_matrices[dataset])[msg_index][train_vocab_lists[dataset].index(word)] += 1
                        (test_bow_matrices[dataset])[msg_index][-1] = 0
                    except:
                        pass
            msg_index += 1
                
                    
    for filename in os.listdir(test_folder + "\\spam"):
        with open(os.path.join(test_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    try:
                        (test_bow_matrices[dataset])[msg_index][train_vocab_lists[dataset].index(word)] += 1
                        (test_bow_matrices[dataset])[msg_index][-1] = 1
                    except:
                        pass
        
            msg_index += 1

print("Using the BAG OF WORD MODEL")
print("The features x examples matrices are: ")

print("Matrix for dataset 1", bow_matrices[0])
print("Matrix for dataset 2", bow_matrices[1])
print("Matrix for dataset 3", bow_matrices[2])


# In[4]:


bnouli_matrices = []

test_bnouli_matrices = []

#iterating over number of messages to create bag of word matrices for all 3 datasets
#Last column of the numpy array is the inference i.e 0 for ham and 1 for spam
for i in range(no_of_datasets):
    bnouli_matrices.append(np.zeros((no_message[i], len(train_vocab_lists[i])+1)))

for i in range(no_of_datasets):
    test_bnouli_matrices.append(np.zeros((no_message[i], len(train_vocab_lists[i])+1)))


for dataset_index, train_folder in enumerate(train_data_folders):
    msg_index = 0
    for filename in os.listdir(train_folder + "\\ham"): 
        with open(os.path.join(train_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    (bnouli_matrices[dataset_index])[msg_index][train_vocab_lists[dataset_index].index(word)] = 1
                    (bnouli_matrices[dataset_index])[msg_index][-1] = 0
            msg_index += 1
                
                    
    for filename in os.listdir(train_folder + "\\spam"):
        with open(os.path.join(train_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    (bnouli_matrices[dataset_index])[msg_index][train_vocab_lists[dataset_index].index(word)] = 1
                    (bnouli_matrices[dataset_index])[msg_index][-1] = 1
            msg_index += 1

for dataset_index, test_folder in enumerate(test_data_folders):
    msg_index = 0
    for filename in os.listdir(test_folder + "\\ham"): 
        with open(os.path.join(test_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    try:
                        (test_bnouli_matrices[dataset_index])[msg_index][train_vocab_lists[dataset_index].index(word)] = 1
                        (test_bnouli_matrices[dataset_index])[msg_index][-1] = 0
                    except:
                        pass
            msg_index += 1
                
                    
    for filename in os.listdir(test_folder + "\\spam"):
        with open(os.path.join(test_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            for sent in f.readlines():
                for word in nltk.tokenize.word_tokenize(sent):
                    try:
                        (test_bnouli_matrices[dataset_index])[msg_index][train_vocab_lists[dataset_index].index(word)] = 1
                        (test_bnouli_matrices[dataset_index])[msg_index][-1] = 1
                    except:
                        pass
            msg_index += 1

    

print("Using the BERNOULLI MODEL")
print("The features x examples matrices are: ")

print("Matrix for dataset 1", bnouli_matrices[0])
print("Matrix for dataset 2", bnouli_matrices[1])
print("Matrix for dataset 3", bnouli_matrices[2])


# In[5]:


for dataset in range(no_of_datasets):
    print("------------------------------dataset: ", dataset + 1, "--------------------------------")
    train_matrices = [bow_matrices, bnouli_matrices]
    test_matrices = [test_bow_matrices, test_bnouli_matrices]
    for i in range(2):
        data_matrix = train_matrices[i][dataset].copy()
        if i == 0:
            print("----------------------------SGD Classifier with BOW MODEL------------------------")
        else:
            print("---------------------------SGD Classifier with Bernoulli Model----------------------")
       
        
        X_train = data_matrix[:,:-1]
        Y_train = data_matrix[:, len(data_matrix[0])-1:]

        Y_train = [y[0] for y in Y_train]   

        test_matrix = test_matrices[i][dataset].copy()

        X_test = test_matrix[:,:-1]
        Y_test = test_matrix[:, len(test_matrix[0])-1:]

        Y_test = [y[0] for y in Y_test]

        sgd = SGDClassifier(eta0=0.1, random_state=42)


        param_grid = {
            'loss': ['log_loss', 'squared_hinge', 'perceptron'],
            'penalty': ['l2', 'l1'],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'optimal', 'adaptive'],
            'max_iter': [100, 250]
        }


        grid_search = GridSearchCV(sgd, param_grid, cv=5, n_jobs=-1, scoring='accuracy')


        grid_search.fit(X_train, Y_train)



        print("Best parameters:", grid_search.best_params_)

        Y_pred = grid_search.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


