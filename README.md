# Misc Machine Learning Projects

## Project Details

### ML Project 1 - SPAM Email Detection
I implemented Naive Bayes and Logistic Regression from scratch to classify emails as HAM (safe) or SPAM. I used Bag of words and Bernoulli model to turn the text data into vectors. After tuning the hyperparameters of Logistic Regression with Bernoulli model, it achieved an accuracy of 98% in classifying emails as SPAM or HAM(safe).

I compared the performance with SGDClassifier and tuned hyperparameters using GridSearchCV from Scikit-learn. The SGDClassifier achieved lesser accuracy than my implementations.   

Read project reports to check the results and model comparison.

### ML Project 2 - Handwritten Digits Recognition 
I implemented Decision Trees and ensemble models including Decision Trees with Bagging, Random Forest Classifier and Gradient Boosting Classifier. I trained the model on synthetic data generated using CNFs with 300-1800 clauses. The Gradient Boosting Classifier achieved the perfect accuracies and Decision Trees with Bagging Achieved similar accuracies.

To classify handwritten digits, I trained the same models on MNIST (https://www.openml.org/d/554) Data with RandomisedSearchCV to tune the Hyperparameters and achieved 97.64 % accuracy on test data with GradientBoostingClassifier. 

Read project reports to check the results and model comparison.

## How to use? 

* Each sub folder is for different projects for different data.
* Each project has its own Readme and project reports.
* Follow Readme instructions to run the particular project.



