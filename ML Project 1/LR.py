#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import os
import math
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('punkt_tab')

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


def train_model(weights, X_train, Y_train, lambda_reg, lr, iterations):
    for iter in range(iterations):
        py1_xi = np.zeros(len(Y_train))
        for i in range(len(X_train)):
            wsum = weights[0] + np.dot(weights[1:], X_train[i]) 
            py1_xi[i] = expit(wsum)
        for i in range(len(weights)):
            if i == 0:
                weights[i] = weights[i]*(1 - lr*lambda_reg)
                for j in range(len(Y_train)):
                    weights[i] += lr*(Y_train[j][0] - py1_xi[j])
            else:
                weights[i] = weights[i]*(1 - lr*lambda_reg)
                for j in range(len(Y_train)):
                    weights[i] += lr*X_train[j][i-1]*(Y_train[j][0] - py1_xi[j])
                
def validate_model(weights, X_val, Y_val, lambda_reg, lr):
    total_preds = len(Y_val)
    correct_preds = 0
    for i in range(len(X_val)):
        wsum = weights[0] + np.dot(weights[1:], X_val[i])

        if wsum > 0 and Y_val[i][0] == 1:
            correct_preds += 1
        elif wsum <= 0 and Y_val[i][0] == 0:
            correct_preds += 1
    
    print("Total values in validation set: ",total_preds)
    print("Correct predictions in validation set: ", correct_preds)
    print("Accuracy when (lambda, learning rate) is", lambda_reg, lr ,": ", correct_preds*100/total_preds)

def test_model(weights, X_val, Y_val):
        Y_test = [y[0] for y in Y_val]
        predictions = [0]*len(Y_test)
        for i in range(len(X_val)):
            wsum = weights[0] + np.dot(weights[1:], X_val[i])
            if wsum > 0:
                predictions[i] = 1
            elif wsum <= 0:
                predictions[i] = 0
        
        accuracy = accuracy_score(Y_test, predictions)
        precision = precision_score(Y_test, predictions)
        recall = recall_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


def experiment_model_hyperparas(lambda_ranges, X_train, Y_train, X_val, Y_val, learning_rates, iter_hard_limits):
    for lambda_reg in lambda_ranges:
        for lr in learning_rates:
            for iterations in iter_hard_limits:
                np.random.seed(31234)
                weights = np.random.uniform(-1, 1, len(train_vocab_lists[dataset])+1)
                print("------------------------------------------------")
                print("Training weights for ", iterations, " iterations.")
                print("----------------------------------------------")
                train_model(weights, X_train, Y_train, lambda_reg, lr, iterations)
                validate_model(weights, X_val, Y_val, lambda_reg, lr)


# In[6]:


for dataset in range(no_of_datasets):
    print("------------------------------dataset: ", dataset + 1, "--------------------------------")
    train_matrices = [bow_matrices, bnouli_matrices]
    test_matrices = [test_bow_matrices, test_bnouli_matrices]
    for i in range(2):
        data_matrix = train_matrices[i][dataset].copy()
        if i == 0:
            print("----------------------------LOR with BOW MODEL------------------------")
        else:
            print("---------------------------LOR with Bernoulli Model----------------------")
        #For the experimentation
        lambda_ranges = [0.01, 0.1, 1, 10]
        learning_rates = [0.01, 0.1, 1.0, 10]
        iter_hard_limits = [10, 20, 50, 100, 250]
        seed_r = 1234

        np.random.seed(seed_r)
        np.random.shuffle(data_matrix)

        partition_70 = int(math.floor(0.7*len(data_matrix)))
        train_set = data_matrix[:partition_70]
        val_set = data_matrix[partition_70:]

        X_train = train_set[:,:-1]
        Y_train = train_set[:, len(train_set[0])-1:]

        X_val = val_set[:,:-1]
        Y_val = val_set[:, len(val_set[0])-1:]


        
        #Just for experimentation (Realised it is like grid-search)
        #Uncomment these to experiment

        """
        lambda_ranges = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
        lr = [0.001, 0.01, 0.1, 1.0, 10]
        iters = [100, 250]
        experiment_model_hyperparas(lambda_ranges, X_train, Y_train, X_val, Y_val, lr, iters)
        """

        #Will take lambda as 0.1, learning rate as 0.1 and 250 iterations, based on the calulations above  

        #The training set for all of the data
        X_train = data_matrix[:,:-1]
        Y_train = data_matrix[:, len(train_set[0])-1:]


        learning_rate = 0.1
        lambda_val = 0.1
        iterations = 250

        weights = np.random.uniform(-1, 1, len(train_vocab_lists[dataset])+1)
        print("------------------------------------------------")
        print("Training final weights all data for dataset : ", dataset + 1, " and ", iterations, " iterations.")
        print("----------------------------------------------")
        train_model(weights, X_train, Y_train, lambda_val, learning_rate, iterations)
        
        
        test_matrix = test_matrices[i][dataset].copy()

        X_test = test_matrix[:,:-1]
        Y_test = test_matrix[:, len(test_matrix[0])-1:]

        test_model(weights, X_test, Y_test)

