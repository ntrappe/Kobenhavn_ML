import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# read in the data
data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
# split input variables and labels
X_train = data_train[:, :-1]
y_train = data_train[:, -1]
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

#------------------------------------------------------------------------#
# EXERCISE 1: Apply a nearest neighbor classifier (1-NN) to the data. 
#------------------------------------------------------------------------#
print('EXERCISE 1')
# given classifier called knn, compute accuracy on test set
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train);
print('Accuracy training: ', accuracy_score(y_train, knn.predict(X_train)))
print('Accuracy testing: ', accuracy_score(y_test, knn.predict(X_test)))
print('--------------------------')
print('EXERCISE 2')

#------------------------------------------------------------------------#
# EXERCISE 2: Apply a nearest neighbor classifier (1-NN) to the data. 
# You are supposed to find a good value for k from {1, 3, 5, 7, 9, 11}. 
# For every choice of k, estimate the performance of the k-NN classifier 
# using 5-fold cross-validation. Pick the k with the lowest average 0-1 
# loss (classification error), which we will call kbest in the following. 
# Only use the training data in the cross-validation process to generate 
# the folds.
#------------------------------------------------------------------------#
def cross_validation(X, y, num_folds, k_vals=[1,3,5,7,9,11]):
    cv_scores = [0,0,0,0,0,0]
    
    # loop over each potential k value
    for i in range(0,len(k_vals)):
        knn_cv = KNeighborsClassifier(n_neighbors=k_vals[i])
    
        # loop over CV folds
        cv_acc_per_k_val = cross_val_score(knn_cv, X, y, cv=num_folds)
        cv_scores[i] = np.mean(cv_acc_per_k_val)

    print('average cv scores for each k val:', cv_scores)
    return k_vals[cv_scores.index(np.amax(cv_scores))]

# determine best k value using 5-fold cross valid
k_best = cross_validation(X_train, y_train, 5)
print('k_best:', k_best)

#------------------------------------------------------------------------#
# EXERCISE 3: Estimate the generalization performance , build a kbest-KNN
# classififer using the complete training data set and evaluate it 
# on the complete independent test set.
#------------------------------------------------------------------------#
# set the model to k best and fit on training, evaluate on testing
knn_best = KNeighborsClassifier(n_neighbors=k_best)
knn_best.fit(X_train, y_train)   # use k-best found in exercise 2
print('--------------------------')
print('EXERCISE 3')
print('K-best NN accuracy score:', accuracy_score(y_test, knn_best.predict(X_test)))

#------------------------------------------------------------------------#
# EXERCISE 4: A basic normalization is to generate zero- mean, unit variance 
# input data. Center and normalize the data and repeat the model selection 
# and classification process from Exercise 2 and Exercise 3. However, keep 
# the general rule from above in mind.
#------------------------------------------------------------------------#
print('--------------------------')
print('EXERCISE 4')
# set the scalar which finds std for a mean of 0 and variance 1
scalar = preprocessing.StandardScaler().fit(X_train)
# normalize training and testing data
X_train_norm = scalar.transform(X_train)
X_test_norm = scalar.transform(X_test)

# determine best k value using 5-fold cross valid ON normalized data
k_best_norm = cross_validation(X_train_norm, y_train, 5)
print('k_best:', k_best_norm)

# set the model to k best and fit on normalized training
# then evaluate on normalized testing
knn_best_norm = KNeighborsClassifier(n_neighbors=k_best_norm)
knn_best_norm.fit(X_train_norm, y_train)
print('K-best NN accuracy score:', accuracy_score(y_test, knn_best_norm.predict(X_test_norm)))
