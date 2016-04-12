'''
    Supervised learning functions
'''

from sklearn.cross_validation import KFold
from sklearn import decomposition, manifold, svm, grid_search, metrics
from sklearn.ensemble import RandomForestClassifier
from decimal import *
import numpy as np
from unsupervised import bagOfWords
from sklearn.linear_model import ElasticNet, ElasticNetCV


def eNetModel(data, labels, nFolds):
    # run SVM with grid search for parameters and leave-one-out cross validation
    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    for train, test in kf:
       data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]

       enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],n_alphas=1000,alphas=[0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])

       enet.fit(data_train, label_train)

       #if label_test == enet.predict(data_test):
       if label_test == 1 and enet.predict(data_test) > 0.5:
           acc += 1
       elif label_test == 0 and enet.predict(data_test) < 0.5:
           acc += 1

    return Decimal(acc)/Decimal(len(data))




def svmModel(data, labels, nFolds):
    #print bagIt
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]

    # run SVM with grid search for parameters and leave-one-out cross validation
    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    for train, test in kf:
    #   scripts = []
    #   documents = {}
    #   for i in train:
    #       scripts.append(bagIt[0][i])
    #       documents[bagIt[0][i]] = bagIt[1][bagIt[0][i]]

       #Traindata, newVocab, featureNames = bagOfWords(scripts, documents, bagIt[2], bagIt[3])
       data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]
       #data_train, data_test, label_train, label_test = Traindata, data[test], labels[train], labels[test]

       svr = svm.SVC()
       clf = grid_search.GridSearchCV(svr, param_grid)
       clf.fit(data_train, label_train)
       #print max(clf.grid_scores_)

       if label_test == clf.predict(data_test):
           acc += 1

       #print label_test, clf.predict(data_test)

    return Decimal(acc)/Decimal(len(data))



def rfModel(data, labels, nFolds, nEstimators):

    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    importances = [[]] * len(data)
    std = [[]] * len(data)
    count = 0
    for train, test in kf:
       data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]

       rf = RandomForestClassifier(n_estimators=nEstimators)
       rf.fit(data_train, label_train)

       if label_test == rf.predict(data_test):
           acc += 1

      # print label_test, rf.predict(data_test)
       importances[count] = rf.feature_importances_
       std[count] = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
       #indices = np.argsort(importances)[::-1]

       count += 1

       #print indices
       #print importances
       #for f in range(data_train.shape[1]):
        #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    importances = np.array(importances)
    importances =  np.mean(importances, axis=0)

    stds = np.array(std)
    stds = np.mean(stds, axis=0)

    return Decimal(acc)/Decimal(len(data)), importances, stds
