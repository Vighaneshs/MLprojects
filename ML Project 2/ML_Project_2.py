#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Please upload the data if not already present.
It should be in the same folder as sample_data.
"""

import zipfile
with zipfile.ZipFile("./project2_data.zip", 'r') as zip_ref:
    zip_ref.extractall("./")


# In[2]:


from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import glob
import os
import re
import itertools


# In[3]:


folder_path = 'all_data'

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

datasets = {}

pattern = r"c\d+_d\d+"

for file in csv_files:
    filename = os.path.basename(file)
    file_match = re.search(pattern, filename)
    if file_match:
        common_index = file_match.group()

        if common_index not in datasets:
            datasets[common_index] = {}

        df = pd.read_csv(file, header=None)

        if filename.startswith('train'):
            datasets[common_index]['train'] = df
        elif filename.startswith('test'):
            datasets[common_index]['test'] = df
        elif filename.startswith('valid'):
            datasets[common_index]['valid'] = df

# for _, data in datasets.items():
#     if 'train' in data:
#         print(f"Train dataset shape: {data['train'].shape}")
#     if 'test' in data:
#         print(f"Test dataset shape: {data['test'].shape}")
#     if 'valid' in data:
#         print(f"Valid dataset shape: {data['valid'].shape}")


# In[4]:


print(datasets.keys())


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

for dataset_name in datasets.keys():

  hyper_para_list = [
      ['gini', 'entropy', 'log_loss'], # Criterion
      ['best', 'random'], # Splitter
      [2, 3, 6, 10, None], # max_depth
      [2, 5, 10, 50], # min_samples_split
      [1, 2, 5, 10], # min_samples_leaf
      ['sqrt', 'log2', None], # max_features
  ]

  best_paras = None
  best_accuracy_validation = 0
  for hype_paras in itertools.product(*hyper_para_list):
    tree_model = DecisionTreeClassifier(criterion=hype_paras[0], splitter=hype_paras[1], max_depth=hype_paras[2],
                                        min_samples_split=hype_paras[3], min_samples_leaf=hype_paras[4],
                                        random_state=0, max_features=hype_paras[5])

    X_train, y_train = datasets[dataset_name]['train'].iloc[:,:-1], datasets[dataset_name]['train'].iloc[:,-1]

    tree_model.fit(X_train, y_train)

    y_validpred = tree_model.predict(datasets[dataset_name]['valid'].iloc[:,:-1])

    accuracy = accuracy_score(datasets[dataset_name]['valid'].iloc[:,-1], y_validpred)
    if accuracy > best_accuracy_validation:
      best_accuracy_validation = accuracy
      best_paras = hype_paras

  tuned_tree_model = DecisionTreeClassifier(criterion=best_paras[0], splitter=best_paras[1], max_depth=best_paras[2],
                                        min_samples_split=best_paras[3], min_samples_leaf=best_paras[4],
                                        random_state=0, max_features=best_paras[5])
  combined_data = pd.concat([datasets[dataset_name]['train'], datasets[dataset_name]['valid']], axis=0)
  X_train, y_train = combined_data.iloc[:,:-1], combined_data.iloc[:,-1]
  X_test, y_test = datasets[dataset_name]['test'].iloc[:,:-1], datasets[dataset_name]['test'].iloc[:,-1]

  tuned_tree_model.fit(X_train, y_train)

  y_pred = tuned_tree_model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  f1 = f1_score(y_test, y_pred, average='binary')

  print("\n\n--------------DECISION TREE CLASSIFIER----------------------")
  print(f"For Dataset - {dataset_name}, best parameter settings found via tuning are :- ")
  print(f"""Criterion - {best_paras[0]}, Splitter - {best_paras[1]},
        Max_Depth - {best_paras[2]}, Min_samples_split - {best_paras[3]},
        Min_samples_leaf - {best_paras[4]}, Max_features - {best_paras[5]}""")
  print(f"Accuracy: {accuracy}")
  print(f"F1 Score: {f1}")
  print("---------------------------------------------------------\n")



# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

for dataset_name in datasets.keys():

  #Based on the previous result, we got the best hyper_parameters for a Decision Tree
  best_dtc_hyper = {
   'c1800_d100':['gini', 'best', 6, 2, 5, None],
   'c1000_d1000':['entropy', 'random', 10, 2, 2, None],
   'c300_d1000':['gini', 'best', 6, 50, 1, None],
   'c1000_d100':['gini', 'random', 6, 2, 2, None],
   'c1800_d5000':['gini', 'best', 10, 2, 2, None],
   'c500_d1000':['entropy', 'random', 10, 50, 1, None],
   'c1800_d1000':['entropy', 'best', 10, 2, 1, None],
   'c1000_d5000':['entropy', 'best', 10, 2, 2, None],
   'c500_d5000':['entropy', 'random', 10, 50, 1, None],
   'c300_d5000':['entropy', 'random', 10, 50, 1, None],
   'c300_d100':['gini', 'random', 2, 2, 1, 'log2'],
   'c500_d100':['entropy', 'best', 6, 2, 5, None],
   'c1500_d100':['entropy', 'best', 6, 2, 5, None],
   'c1500_d1000':['entropy', 'best', 10, 2, 2, None],
   'c1500_d5000':['entropy', 'best', 10, 2, 2, None]
  }

  bc_hyper_para_list = [
      [10, 50, 100], # n_estimators
      [0.5, 1.0], # max_samples
      [0.5, 1.0], # max_features
      [True], # bootstrap
      [False, True], # bootstrap_features
      [False], # oob_score
      [False]  # warm_start
  ]

  best_paras = None
  best_accuracy_validation = 0
  for hype_paras_bagging in itertools.product(*bc_hyper_para_list):
    hype_paras = best_dtc_hyper[dataset_name]
    dc_tree_model = DecisionTreeClassifier(criterion=hype_paras[0], splitter=hype_paras[1], max_depth=hype_paras[2],
                                        min_samples_split=hype_paras[3], min_samples_leaf=hype_paras[4],
                                        random_state=0, max_features=hype_paras[5])

    tree_model = BaggingClassifier(estimator=dc_tree_model, n_estimators=hype_paras_bagging[0], random_state=0,
                                   max_samples=hype_paras_bagging[1], max_features=hype_paras_bagging[2],
                                   bootstrap=hype_paras_bagging[3], bootstrap_features=hype_paras_bagging[4],
                                   oob_score=hype_paras_bagging[5], warm_start=hype_paras_bagging[6])

    X_train, y_train = datasets[dataset_name]['train'].iloc[:,:-1], datasets[dataset_name]['train'].iloc[:,-1]

    tree_model.fit(X_train, y_train)

    y_validpred = tree_model.predict(datasets[dataset_name]['valid'].iloc[:,:-1])

    accuracy = accuracy_score(datasets[dataset_name]['valid'].iloc[:,-1], y_validpred)
    if accuracy > best_accuracy_validation:
      best_accuracy_validation = accuracy
      best_paras = hype_paras_bagging

  hype_paras = best_dtc_hyper[dataset_name]
  dc_tree_model = DecisionTreeClassifier(criterion=hype_paras[0], splitter=hype_paras[1], max_depth=hype_paras[2],
                                        min_samples_split=hype_paras[3], min_samples_leaf=hype_paras[4],
                                        random_state=0, max_features=hype_paras[5])
  hype_paras_bagging = best_paras
  tuned_tree_model = BaggingClassifier(estimator=dc_tree_model, n_estimators=hype_paras_bagging[0], random_state=0,
                                   max_samples=hype_paras_bagging[1], max_features=hype_paras_bagging[2],
                                   bootstrap=hype_paras_bagging[3], bootstrap_features=hype_paras_bagging[4],
                                   oob_score=hype_paras_bagging[5], warm_start=hype_paras_bagging[6])

  combined_data = pd.concat([datasets[dataset_name]['train'], datasets[dataset_name]['valid']], axis=0)
  X_train, y_train = combined_data.iloc[:,:-1], combined_data.iloc[:,-1]
  X_test, y_test = datasets[dataset_name]['test'].iloc[:,:-1], datasets[dataset_name]['test'].iloc[:,-1]

  tuned_tree_model.fit(X_train, y_train)

  y_pred = tuned_tree_model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  f1 = f1_score(y_test, y_pred, average='binary')

  print("\n\n---------------DECISION TREE WITH BAGGING---------------------------")
  print(f"For Dataset - {dataset_name}, best parameter settings found via tuning for the Decision Tree Classifier are :- ")
  print(f"""Criterion - {hype_paras[0]}, Splitter - {hype_paras[1]},
        Max_Depth - {hype_paras[2]}, Min_samples_split - {hype_paras[3]},
        Min_samples_leaf - {hype_paras[4]}, Max_features - {hype_paras[5]}\n""")

  print(f"For Dataset - {dataset_name}, best parameter settings found via tuning for the Bagging Classifier are :- ")
  print(f"""no of estimators = {best_paras[0]}, max_samples = {best_paras[1]},
            max_features = {best_paras[2]}, bootstrap? = {best_paras[3]},
            bootstrap_features? = {best_paras[4]}, oob_score? = {best_paras[5]},
            warm_start? = {best_paras[6]}""")

  print(f"Accuracy: {accuracy}")
  print(f"F1 Score: {f1}")
  print("---------------------------------------------------------\n")



# In[7]:


from sklearn.ensemble import RandomForestClassifier

for dataset_name in datasets.keys():

  hyper_para_list = [
      ['gini', 'entropy', 'log_loss'], # Criterion
      [0], # verbose
      [2, 6, 10], # max_depth
      [2, 10], # min_samples_split
      [1, 2, 5], # min_samples_leaf
      [None], # max_features
      [50, 100], # n_estimators
  ]

  best_paras = None
  best_accuracy_validation = 0
  for hype_paras in itertools.product(*hyper_para_list):
    tree_model = RandomForestClassifier(n_jobs=-1, criterion=hype_paras[0], max_depth=hype_paras[2],
                                        min_samples_split=hype_paras[3], min_samples_leaf=hype_paras[4],
                                        random_state=0, max_features=hype_paras[5], n_estimators=hype_paras[6], verbose=hype_paras[1])

    X_train, y_train = datasets[dataset_name]['train'].iloc[:,:-1], datasets[dataset_name]['train'].iloc[:,-1]

    tree_model.fit(X_train, y_train)

    y_validpred = tree_model.predict(datasets[dataset_name]['valid'].iloc[:,:-1])

    accuracy = accuracy_score(datasets[dataset_name]['valid'].iloc[:,-1], y_validpred)
    if accuracy > best_accuracy_validation:
      best_accuracy_validation = accuracy
      best_paras = hype_paras

  tuned_tree_model = RandomForestClassifier(n_jobs=-1, criterion=best_paras[0], max_depth=best_paras[2],
                                        min_samples_split=best_paras[3], min_samples_leaf=best_paras[4],
                                        random_state=0, max_features=best_paras[5], n_estimators=best_paras[6], verbose=best_paras[1])
  combined_data = pd.concat([datasets[dataset_name]['train'], datasets[dataset_name]['valid']], axis=0)
  X_train, y_train = combined_data.iloc[:,:-1], combined_data.iloc[:,-1]
  X_test, y_test = datasets[dataset_name]['test'].iloc[:,:-1], datasets[dataset_name]['test'].iloc[:,-1]

  tuned_tree_model.fit(X_train, y_train)

  y_pred = tuned_tree_model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  f1 = f1_score(y_test, y_pred, average='binary')

  print("\n\n--------------RANDOM FOREST CLASSIFIER----------------------")
  print(f"For Dataset - {dataset_name}, best parameter settings found via tuning are :- ")
  print(f"""Criterion - {best_paras[0]}, Max_Depth - {best_paras[2]}, Min_samples_split - {best_paras[3]},
        Min_samples_leaf - {best_paras[4]}, Max_features - {best_paras[5]},
        n_estimators - {best_paras[6]}""")
  print(f"Accuracy: {accuracy}")
  print(f"F1 Score: {f1}")
  print("---------------------------------------------------------\n")


# In[6]:


from sklearn.ensemble import GradientBoostingClassifier

for dataset_name in datasets.keys():

  hyper_para_list = [
      ['exponential', 'log_loss'], # loss
      [0.1], # learning_rate
      [2, 6, 10], # max_depth
      [2, 10], # min_samples_split
      [1, 2, 5], # min_samples_leaf
      [None], # max_features
      [50, 100], # n_estimators
      [0.5, 1.0], # subsample
      ['friedman_mse', 'squared_error'], # criterion
  ]

  best_paras = None
  best_accuracy_validation = 0
  for hype_paras in itertools.product(*hyper_para_list):
    tree_model = GradientBoostingClassifier(loss=hype_paras[0], learning_rate=hype_paras[1], max_depth=hype_paras[2],
                                        min_samples_split=hype_paras[3], min_samples_leaf=hype_paras[4],
                                        random_state=0, max_features=hype_paras[5], n_estimators=hype_paras[6], subsample=hype_paras[7], criterion=hype_paras[8], verbose=0)

    X_train, y_train = datasets[dataset_name]['train'].iloc[:,:-1], datasets[dataset_name]['train'].iloc[:,-1]

    tree_model.fit(X_train, y_train)

    y_validpred = tree_model.predict(datasets[dataset_name]['valid'].iloc[:,:-1])

    accuracy = accuracy_score(datasets[dataset_name]['valid'].iloc[:,-1], y_validpred)
    if accuracy > best_accuracy_validation:
      best_accuracy_validation = accuracy
      best_paras = hype_paras

  tuned_tree_model = GradientBoostingClassifier(loss=best_paras[0], learning_rate=best_paras[1], max_depth=best_paras[2],
                                        min_samples_split=best_paras[3], min_samples_leaf=best_paras[4],
                                        random_state=0, max_features=best_paras[5], n_estimators=best_paras[6], subsample=best_paras[7], criterion=best_paras[8], verbose=0)


  combined_data = pd.concat([datasets[dataset_name]['train'], datasets[dataset_name]['valid']], axis=0)
  X_train, y_train = combined_data.iloc[:,:-1], combined_data.iloc[:,-1]
  X_test, y_test = datasets[dataset_name]['test'].iloc[:,:-1], datasets[dataset_name]['test'].iloc[:,-1]

  tuned_tree_model.fit(X_train, y_train)

  y_pred = tuned_tree_model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  f1 = f1_score(y_test, y_pred, average='binary')



  print("\n\n--------------GRADIENT BOOSTING CLASSIFIER----------------------")
  print(f"For Dataset - {dataset_name}, best parameter settings found via tuning are :- ")
  print(f"""Loss - {best_paras[0]}, Learning Rate - {best_paras[1]}
        Criterion - {best_paras[8]}, Max_Depth - {best_paras[2]}, Min_samples_split - {best_paras[3]},
        Min_samples_leaf - {best_paras[4]}, Max_features - {best_paras[5]},
        n_estimators - {best_paras[6]}, subsample - {best_paras[7]}""")
  print(f"Accuracy: {accuracy}")
  print(f"F1 Score: {f1}")
  print("---------------------------------------------------------\n")


# 

# In[3]:


"""
MINST 
"""
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# In[15]:


from sklearn.tree import DecisionTreeClassifier

#chosing the hyper-parameters which were chosen as the best on average for previous DecisionTree Model

tree_model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10,
                                      min_samples_split=2, min_samples_leaf=2,
                                      random_state=0, max_features=None)

tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n\n--------------DECISION TREE CLASSIFIER On MNIST----------------------")
print(f"Accuracy: {accuracy}")
print("---------------------------------------------------------\n")



# In[21]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

#chosing the hyper-parameters which were chosen as the best on average for previous Bagging with DecisionTree Model

dc_tree_model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10,
                                      min_samples_split=2, min_samples_leaf=2,
                                      random_state=0, max_features=None)

tree_model = BaggingClassifier(estimator=dc_tree_model, n_estimators=200, random_state=0,
                                   max_samples=1.0, max_features=0.5)

tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n\n--------------DECISION TREE CLASSIFIER with Bagging On MNIST----------------------")
print(f"Accuracy: {accuracy}")
print("---------------------------------------------------------\n")



# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


#chosing the hyper-parameters which were chosen as the best on average for previous RandomForestClassifier Model

tree_model = RandomForestClassifier(verbose=0, random_state=0)

params = {
    'n_estimators': randint(60, 200),       
    'max_depth': randint(3, 10),                     
    'min_samples_split': randint(2, 10),     
    'min_samples_leaf': randint(1, 10),      
    'max_features': ['sqrt', 'log2', None]  
}


rand_s = RandomizedSearchCV(
    estimator=tree_model, 
    param_distributions=params, 
    n_iter=100,     
    cv=5,               
    n_jobs=-1,      
    random_state=42
)

rand_s.fit(X_train, y_train)

best_model = rand_s.best_estimator_
test_score = best_model.score(X_test, y_test)





# In[5]:


print("\n\n--------------RANDOM FOREST CLASSIFIER On MNIST----------------------")
print(f"Accuracy: {test_score}")
print("---------------------------------------------------------\n")


# In[8]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

#chosing the hyper-parameters which were chosen as the best on average for previous GradientBoostingClassifier Model

tree_model = GradientBoostingClassifier(loss='log_loss', criterion='friedman_mse', verbose=0)



params = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.5, 1.0),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=tree_model, 
    param_distributions=params, 
    n_iter=20, 
    cv=5, 
    verbose=1,  
    random_state=0
)


rand_s.fit(X_train, y_train)

best_model = rand_s.best_estimator_
test_score = best_model.score(X_test, y_test)

print("\n\n--------------Gradient Boosting CLASSIFIER On MNIST----------------------")
print(f"Accuracy: {test_score}")
print("---------------------------------------------------------\n")

