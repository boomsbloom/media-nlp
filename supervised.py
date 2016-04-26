'''
    Supervised learning functions
'''

from sklearn.cross_validation import KFold
from sklearn import decomposition, manifold, svm, grid_search, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from decimal import *
import numpy as np
from operator import itemgetter
from unsupervised import bagOfWords
from sklearn.linear_model import ElasticNet, ElasticNetCV

nFeats = 300

def featureSelection(train_text, train_docs, label_train, nFeats):
    # select by word frequency in training set and return only top n words
    data_train, newVocab, feats_train = bagOfWords(train_text, train_docs, True, 0, False, False)

    # counts = np.sum(data_train, axis=0)
    # sorted_feats = [x for (y,x) in sorted(zip(counts,feats_train),reverse=True)]

    # train a rf on the training set and use the top n important features as features for loocv
    # rf = RandomForestClassifier(n_estimators=1000)
    # rf.fit(data_train, label_train)
    # importances = rf.feature_importances_
    # importances = np.array(importances)
    # sorted_feats = [x for (y,x) in sorted(zip(importances,feats_train),reverse=True)]

    # hack to turn off feature selection...
    sorted_feats = feats_train

    return sorted_feats[:nFeats]


def sortBySelected(data_list, selected_feats, featureNames):
    new_list = [[]] * len(data_list)
    for t in range(len(data_list)):
        data_t = []
        for f in range(len(featureNames)):
            if featureNames[f] in selected_feats:
                data_t.append(data_list[t][f])
        new_list[t] = data_t
    return new_list

def getSelectedFeatures(train, test, texts, featureNames, documents, train_label, nFeats):
    train_text = list(itemgetter(*train)(texts))
    test_text = [itemgetter(*test)(texts)]

    train_docs = {}
    for txt in train_text:
        train_docs[txt] = documents[txt]

    selected_feats = featureSelection(train_text, train_docs, train_label, nFeats)

    return selected_feats

def eNetModel(data, labels, featureNames, texts, documents, nFolds):
    # run SVM with grid search for parameters and leave-one-out cross validation
    kf = KFold(len(texts), n_folds=nFolds)
    acc = 0
    mean_coefs = []
    for train, test in kf:

        # test_docs = {}
        label_train = labels[train]
        selected_feats = getSelectedFeatures(train, test, texts, featureNames, documents, label_train, nFeats)

        full_train_data, full_test_data, label_train, label_test = data[train], data[test], labels[train], labels[test]

        data_train = sortBySelected(full_train_data, selected_feats, featureNames)
        data_test = sortBySelected(full_test_data, selected_feats, featureNames)

        enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],n_alphas=1000,alphas=[0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])

        enet.fit(data_train, label_train)

        data_train = np.asarray(data_train,dtype=float)
        label_train = np.asarray(label_train,dtype=float)

        vals = enet.path(data_train, label_train)
        mean_coefs.append(np.mean(vals[1],axis=1))

        if label_test == 1 and enet.predict(data_test) > 0.5:
            acc += 1
        elif label_test == 0 and enet.predict(data_test) < 0.5:
            acc += 1

        if len(mean_coefs) % 10 == 0:
            print str(len(mean_coefs)), 'out of %s subs finished' %(str(len(data)))

    mean_coefs = np.mean(np.array(mean_coefs), axis=0)

    return Decimal(acc)/Decimal(len(data)), mean_coefs

def svmModel(data, labels, featureNames, texts, documents, nFolds):
    #print bagIt
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]

    # run SVM with grid search for parameters and leave-one-out cross validation
    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    for train, test in kf:

       #data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]
       #data_train, data_test, label_train, label_test = Traindata, data[test], labels[train], labels[test]
       label_train = labels[train]
       selected_feats = getSelectedFeatures(train, test, texts, featureNames, documents, label_train, nFeats)

       full_train_data, full_test_data, label_train, label_test = data[train], data[test], labels[train], labels[test]

       data_train = sortBySelected(full_train_data, selected_feats, featureNames)
       data_test = sortBySelected(full_test_data, selected_feats, featureNames)

       svr = svm.SVC()
       clf = grid_search.GridSearchCV(svr, param_grid)
       clf.fit(data_train, label_train)
       #print max(clf.grid_scores_)

       if label_test == clf.predict(data_test):
           acc += 1

       #print label_test, clf.predict(data_test)

    return Decimal(acc)/Decimal(len(data))



def rfModel(data, labels, featureNames, texts, documents, nFolds, nEstimators):

    kf = KFold(len(data), n_folds=nFolds)
    acc = 0
    importances = [[]] * len(data)
    std = [[]] * len(data)
    count = 0
    for train, test in kf:
       #data_train, data_test, label_train, label_test = data[train], data[test], labels[train], labels[test]
       label_train = labels[train]
       selected_feats = getSelectedFeatures(train, test, texts, featureNames, documents, label_train, nFeats)

       full_train_data, full_test_data, label_train, label_test = data[train], data[test], labels[train], labels[test]

       data_train = sortBySelected(full_train_data, selected_feats, featureNames)
       data_test = sortBySelected(full_test_data, selected_feats, featureNames)

       rf = RandomForestClassifier(n_estimators=nEstimators)
       rf.fit(data_train, label_train)

       if label_test == rf.predict(data_test):
           acc += 1

       #print label_test, rf.predict(data_test)
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
