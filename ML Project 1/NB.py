#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import nltk
import os
import math
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
nltk.download('punkt_tab')

no_of_datasets = 3


# In[16]:

print("Data Processing in progress........\n")
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


# In[17]:


bow_matrices = []

#iterating over number of messages to create bag of word matrices for all 3 datasets
#Last column of the numpy array is the inference i.e 0 for ham and 1 for spam
for i in range(no_of_datasets):
    bow_matrices.append(np.zeros((no_message[i], len(train_vocab_lists[i])+1)))


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

print("Using the BAG OF WORD MODEL")
print("The features x examples matrices are: ")

print("Matrix for dataset 1", bow_matrices[0])
print("Matrix for dataset 2", bow_matrices[1])
print("Matrix for dataset 3", bow_matrices[2])


# In[18]:


bnouli_matrices = []

#iterating over number of messages to create bag of word matrices for all 3 datasets
#Last column of the numpy array is the inference i.e 0 for ham and 1 for spam
for i in range(no_of_datasets):
    bnouli_matrices.append(np.zeros((no_message[i], len(train_vocab_lists[i])+1)))


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

print("Using the BERNOULLI MODEL")
print("The features x examples matrices are: ")

print("Matrix for dataset 1", bnouli_matrices[0])
print("Matrix for dataset 2", bnouli_matrices[1])
print("Matrix for dataset 3", bnouli_matrices[2])


# In[19]:


#Training multinomial Naive Bayes Model
#We have already extracted Vocabulary = train_vocab_list and N = no_message
class_types = [0, 1]


#Precomputing stuff for training
#No of emails in each class
message_in_class_count = []

for dataset, train_folder in enumerate(train_data_folders):
    msg_cnt = [0,0]
    #Counting ham messages
    for filename in os.listdir(train_folder + "\\ham"): 
        with open(os.path.join(train_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            msg_cnt[0] += 1
                
    #Counting Spam messages                    
    for filename in os.listdir(train_folder + "\\spam"):
        with open(os.path.join(train_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            msg_cnt[1] += 1
    message_in_class_count.append(msg_cnt)


prior = [[0.0, 0.0] for i in range(no_of_datasets)]


#The weight of the probabilities
cond_probability = [[[0.0 for k in range(len(train_vocab_lists[i]))] for _ in range(len(class_types))] for i in range(no_of_datasets)]

print("Training in progress...............\n")

#Training the model
for i in range(no_of_datasets):
    for c in class_types:
        #prior probability i is dataset index and c is class
        prior[i][c] = message_in_class_count[i][c]/no_message[i]
        tsum = 0.0
        tt = [[0.0 for k in range(len(train_vocab_lists[i]))] for _ in range(len(class_types))]

        tsum = 0.0
        for word_index in range(len(train_vocab_lists[i])):
            for wc in range(len(bow_matrices[i])):
                if(bow_matrices[i][wc][-1] == c):
                    tt[c][word_index] += bow_matrices[i][wc][word_index]
                    tsum += bow_matrices[i][wc][word_index]
        
        for word_index in range(len(train_vocab_lists[i])):    
            cond_probability[i][c][word_index] = (tt[c][word_index] + 0.5)/(tsum + len(train_vocab_lists[i]))



# In[20]:


#Applying the model and calculating scores of each email and actual values

score = [[[] for _ in class_types] for _ in range(no_of_datasets)]
actual = [[] for _ in range(no_of_datasets)]
for dataset, test_folder in enumerate(test_data_folders):
    for filename in os.listdir(test_folder + "\\ham"):
        with open(os.path.join(test_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            file_content = f.readlines()
            actual[dataset].append(0)
            for c in class_types:
                tscore = math.log(prior[dataset][c])
                for sent in file_content:
                    for word in nltk.tokenize.word_tokenize(sent):
                        index = -1
                        try:
                            index = train_vocab_lists[dataset].index(word)
                        except:
                            continue
                        finally:
                            tscore += math.log(cond_probability[dataset][c][index])
                score[dataset][c].append(tscore)
                    
                    
    for filename in os.listdir(test_folder + "\\spam"):
        with open(os.path.join(test_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            file_content = f.readlines()
            actual[dataset].append(1)
            for c in class_types:
                tscore = math.log(prior[dataset][c])
                for sent in file_content:
                    for word in nltk.tokenize.word_tokenize(sent):
                        try:
                            index = train_vocab_lists[dataset].index(word)
                        except:
                            continue
                        else:
                            tscore += math.log(cond_probability[dataset][c][index])
                score[dataset][c].append(tscore)


# In[21]:


predictions = [[] for _ in range(no_of_datasets)]
for dataset in range(no_of_datasets):
    for i in range(len(actual[dataset])):
        predict = -1
        if score[dataset][0][i] <= score[dataset][1][i]:
            predict = 1 #It is spam
        else:
            predict = 0 #It is ham
            
        predictions[dataset].append(predict)
    
    accuracy = accuracy_score(actual[dataset], predictions[dataset])
    precision = precision_score(actual[dataset], predictions[dataset])
    recall = recall_score(actual[dataset], predictions[dataset])
    f1 = f1_score(actual[dataset], predictions[dataset])

    print("---------------------------------- For Naive Bayes Multinomial model (BOW)-----------------------------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("---------------------------------------------------------------------------")
         


# In[22]:
time.sleep(1)
print("\n\n\n")

#Training discrete Naive Bayes discrete Model
#We have already extracted Vocabulary = train_vocab_list and N = no_message
class_types = [0, 1]


#Precomputing stuff for training
#No of emails in each class
message_in_class_count = []

for dataset, train_folder in enumerate(train_data_folders):
    msg_cnt = [0,0]
    #Counting ham messages
    for filename in os.listdir(train_folder + "\\ham"): 
        with open(os.path.join(train_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            msg_cnt[0] += 1
                
    #Counting Spam messages                    
    for filename in os.listdir(train_folder + "\\spam"):
        with open(os.path.join(train_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            msg_cnt[1] += 1
    message_in_class_count.append(msg_cnt)


prior = [[0.0, 0.0] for i in range(no_of_datasets)]


#The weight of the probabilities
cond_probability = [[[0.0 for k in range(len(train_vocab_lists[i]))] for _ in range(len(class_types))] for i in range(no_of_datasets)]

#Training the model
for i in range(no_of_datasets):
    for c in class_types:
        #prior probability i is dataset index and c is class
        prior[i][c] = message_in_class_count[i][c]/no_message[i]
        tt = [[0.0 for k in range(len(train_vocab_lists[i]))] for _ in range(len(class_types))]

        for word_index in range(len(train_vocab_lists[i])):
            for wc in range(len(bnouli_matrices[i])):
                if(bnouli_matrices[i][wc][-1] == c):
                    tt[c][word_index] += bnouli_matrices[i][wc][word_index]
    
        
        for word_index in range(len(train_vocab_lists[i])):    
            cond_probability[i][c][word_index] = (tt[c][word_index] + 0.5)/(message_in_class_count[i][c] + 1.0)


# In[23]:


#Applying the model and calculating scores of each email and actual values

#Calculating the sum of log(1-condprob[t][c]) for all the datasets
#For a word existing in document, I can subtract log(1-condprob[t][c]) and add condprob[t][c]
init_scores = [[0 for _ in class_types] for i in range(no_of_datasets)]
for dataset in range(no_of_datasets):
    for c in class_types:
        for word_index in range(len(train_vocab_lists[dataset])):
            init_scores[dataset][c] += math.log(1-cond_probability[dataset][c][word_index])

score = [[[] for _ in class_types] for i in range(no_of_datasets)]
actual = [[] for _ in range(no_of_datasets)]
for dataset, test_folder in enumerate(test_data_folders):
    for filename in os.listdir(test_folder + "\\ham"):
        with open(os.path.join(test_folder + "\\ham\\" + filename), 'r', errors='ignore') as f:
            file_content = f.readlines()
            actual[dataset].append(0)
            for c in class_types:
                tscore = math.log(prior[dataset][c]) + init_scores[dataset][c]
                for sent in file_content:
                    for word in nltk.tokenize.word_tokenize(sent):
                        index = -1
                        try:
                            index = train_vocab_lists[dataset].index(word)
                        except:
                            continue
                        finally:
                            tscore += math.log(cond_probability[dataset][c][index])
                            tscore -= math.log(1-cond_probability[dataset][c][index])
                score[dataset][c].append(tscore)
                    
                    
    for filename in os.listdir(test_folder + "\\spam"):
        with open(os.path.join(test_folder + "\\spam\\" + filename), 'r', errors='ignore') as f:
            file_content = f.readlines()
            actual[dataset].append(1)
            for c in class_types:
                tscore = math.log(prior[dataset][c]) + init_scores[dataset][c]
                for sent in file_content:
                    for word in nltk.tokenize.word_tokenize(sent):
                        try:
                            index = train_vocab_lists[dataset].index(word)
                        except:
                            continue
                        else:
                            tscore += math.log(cond_probability[dataset][c][index])
                            tscore -= math.log(1-cond_probability[dataset][c][index])
                score[dataset][c].append(tscore)



# In[24]:


predictions = [[] for _ in range(no_of_datasets)]
for dataset in range(no_of_datasets):
    for i in range(len(actual[dataset])):
        predict = -1
        if score[dataset][0][i] <= score[dataset][1][i]:
            predict = 1 #It is spam
        else:
            predict = 0 #It is ham
            
        predictions[dataset].append(predict)
    
    accuracy = accuracy_score(actual[dataset], predictions[dataset])
    precision = precision_score(actual[dataset], predictions[dataset])
    recall = recall_score(actual[dataset], predictions[dataset])
    f1 = f1_score(actual[dataset], predictions[dataset])

    print("------------------------------ For the Naive Bayes Discrete Model (Bernoulli)---------------------------------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("---------------------------------------------------------------------------")

