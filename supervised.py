'''
    Supervised learning functions
'''

from sklearn.cross_validation import KFold
from sklearn import decomposition, manifold, svm, grid_search
from sklearn.ensemble import RandomForestClassifier
from decimal import *

def svmModel(data, labels, nFolds):
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]

    # run SVM with grid search for parameters and leave-one-out cross validation
    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    for train, test in kf:
       data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]

       svr = svm.SVC()
       clf = grid_search.GridSearchCV(svr, param_grid)
       clf.fit(data_train, label_train)

       if label_test == clf.predict(data_test):
           acc += 1

       #print label_test, clf.predict(data_test)

    return Decimal(acc)/Decimal(len(data))



def rfModel(data, labels, nFolds):

    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    for train, test in kf:
       data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]

       rf = RandomForestClassifier(n_estimators=100)
       rf.fit(data_train, label_train)

       if label_test == rf.predict(data_test):
           acc += 1

       #print label_test, rf.predict(data_test)

    return Decimal(acc)/Decimal(len(data))
