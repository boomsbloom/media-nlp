from __future__ import division
import os
import numpy as np
from processing import getDocuments
from unsupervised import bagOfWords
from sklearn.cross_validation import KFold
from sklearn import decomposition, manifold, svm, grid_search, metrics
from sklearn.ensemble import RandomForestClassifier

groups = ["AD","TD"]

# paths = ['texts/multiple_sites_half/NYU/AD_4',
#         'texts/multiple_sites_half/NYU/TD_4',
#         'texts/multiple_sites_half/OHSU/AD_4',
#         'texts/multiple_sites_half/OHSU/TD_4',
#         'texts/multiple_sites_half/PKU/AD_4',
#         'texts/multiple_sites_half/PKU/TD_4']

paths = ['texts/multiple_sites_full_2letter/NYU/AD_2',
        'texts/multiple_sites_full_2letter/NYU/TD_2',
        'texts/multiple_sites_full_2letter/OHSU/AD_2',
        'texts/multiple_sites_full_2letter/OHSU/TD_2',
        'texts/multiple_sites_full_2letter/PKU/AD_2',
        'texts/multiple_sites_full_2letter/PKU/TD_2']

train_paths = {'AD':['texts/multiple_sites_full_2letter/PKU/AD_2'],'TD':['texts/multiple_sites_full_2letter/PKU/TD_2']};
#test_paths = {'AD':['texts/multiple_sites_full/OHSU/AD_4','texts/multiple_sites_full/PKU/AD_4'],
#           'TD':['texts/multiple_sites_full/OHSU/TD_4','texts/multiple_sites_full/PKU/TD_4']};
test_paths = {'AD':['texts/multiple_sites_full_2letter/NYU/AD_2'],
             'TD':['texts/multiple_sites_full_2letter/NYU/TD_2']};

#### load and process data ####

def loadData(paths):
    documents, texts = [{"AD":[],"TD":[]} for _ in range(2)]
    for group in groups:
        for ds in range(len(paths['AD'])):
            currTexts = sorted([os.path.join(paths[group][ds], fn) for fn in os.listdir(paths[group][ds])])
            for text in currTexts:
                if '.DS_Store' in text:
                    currTexts.remove(text)
            texts[group].append(currTexts)
        texts[group] = sum(texts[group], [])
        documents[group] = getDocuments(texts[group], 'none', False, texts[group])

    return texts, documents

test_texts, test_docs = loadData(test_paths)
train_texts, train_docs = loadData(train_paths)

full_texts = test_texts["AD"] + train_texts["AD"] + test_texts["TD"] + train_texts["TD"]
full_docs = test_docs["AD"]
full_docs.update(train_docs["AD"])
full_docs.update(test_docs["TD"])
full_docs.update(train_docs["TD"])

counts, reduced, words = bagOfWords(full_texts,full_docs, True, 0, False, False)

AD_test = counts[0:len(test_texts["AD"])]
AD_train = counts[len(test_texts["AD"]):len(test_texts["AD"])+len(train_texts["AD"])]
currInd = len(test_texts["AD"])+len(train_texts["AD"])
TD_test = counts[currInd:currInd+len(test_texts["TD"])]
currInd = currInd+len(test_texts["TD"])
TD_train = counts[currInd:currInd+len(train_texts["TD"])]

train = np.concatenate((AD_train, TD_train))
train_label = np.asarray([0] * len(AD_train) + [1] * len(TD_train))
test = np.concatenate((AD_test, TD_test))
test_label = np.asarray([0] * len(AD_test) + [1] * len(TD_test))

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(train, train_label)

predictions = rf.predict(test)

correct = 0
for i in range(len(test_label)):
    if test_label[i] == predictions[i]:
        correct += 1

acc = correct/len(predictions)
print acc, 'correct'












#
