import os, operator, scipy, csv, json
import functions as f
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from processing import getDocuments
from contexts import getnGrams
from occurrences import *
from supervised import *
from unsupervised import *


isCorpus = False
delimiter = 'none'

nGramsinCorpus = False
windowGrams = False
gramsOnly = False
mincount = 4

nModels = 5

nLabelOne = 40 #number of TDs
nLabelTwo = 40 #number of ADs
labels  = np.asarray([0] * nLabelOne + [1] * nLabelTwo)
nFolds = len(labels) #leave-one-out
nEstimators = 1000 #1000 #number of estimators for random forest classifier

rf_accs = {}
for length in range(2):
    if length == 0:
        pre_path = 'texts/ADHD_various_letters_half/'
    else:
        pre_path = 'texts/ADHD_various_letters_full/'

    let_acc = [0] * 14

    for let in range(2,16):
        path = pre_path + '%s_word'%(let)

        textNames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
        scripts = textNames

        new_texts = []
        for script in scripts:
            if '.DS_Store' not in script:
                new_texts.append(script)

        documents = getDocuments(new_texts, delimiter, isCorpus, textNames)

        data, newVocab, featureNames = bagOfWords(new_texts, documents, nGramsinCorpus, mincount, windowGrams, gramsOnly)

        rfACC = [0] * nModels
        importances = [[]] * nModels
        stds = [[]] * nModels

        for i in range(nModels):
           print "Running RF with %i estimators...\n" %(nEstimators)
           rfACC[i], importances[i], stds[i] = rfModel(data, labels, nFolds, nEstimators)
           idx = (-importances[i]).argsort()[:10]

           print "Top 10 features:"
           for j in idx:
               print (featureNames[j], importances[i][j]), "std: ", stds[i][j]

           print "\nrf ACC:", rfACC[i], "\n"

           rfACC[i] = float(rfACC[i])

        let_acc[let-2] = np.mean(rfACC)

    print rf_accs
    rf_accs[pre_path] = let_acc

with open('rf_accs_4mincount.json', 'w') as fp:
    json.dump(rf_accs, fp)
