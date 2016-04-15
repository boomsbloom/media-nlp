import os, operator, scipy, csv
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
#from plotter import *
#from unsupervised import ldaModel

#################################################
################ LOAD INPUT DATA ################
#################################################

#path = 'texts/multiple_sites_half/PKU/both'
#path = 'texts/multiple_sites_full/OHSU/both'
path = 'texts/multiple_sites_half/all_data'
#path = 'texts/multiple_sites_half/NYU_and_PKU'
#path = 'texts/multiple_sites_half/NYU/both'
#path = 'texts/multiple_sites_full_2letter/OHSU/both'
#path = 'texts/multiple_sites_full_2letter/all_data'
#path = 'texts/multiple_sites_full_2letter/NYU_PKU'
#path = 'texts/ADHD_various_half/2_word/'

title = "TD letter counts"
#letters = ['a','b','c','d','e']
letters = ['b','c']
#path = 'texts/TD_4_half'
textNames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

# choose whether input is one document with your whole corpus in it (to be split up)
# ..or whether input is multiple documents
#isCorpus = True
isCorpus = False

if isCorpus:
    #scripts = 'texts/full_4letter_phrase_corpus.txt'
    scripts = 'topicalPhrases/rawFiles/4letters_full_corpus.txt'
else:
    scripts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
    for script in scripts:
        if '.DS_Store' in script:
            scripts.remove(script)

#################################################
############### CHOOSE PARAMETERS ###############
#################################################

network_wise = False

nModels = 10 # number of times you want modeling to run
nGrams = 10 # number of words in context ..only if running context calculation

# for LDA
runLDA = False # whether to run LDA
delimiter = 'none' #or ',' type of delimiter between your words in the document
nTopics = 20 # number of topics to create
nWords = 10 #4 # number of words per topic; is actually n - 1 (so 3 for 2 words)
nIters = 1000 # number of iterations for sampling

# for HDP
runHDP = False  # whether to run HDP
tLimit = 150#150 # limit on number of topics to look for (default is 150)
# note: larger the limit on topics the more sparse the classification matrix

# for DTM
runDTM = False
#nTopics = 10 #20
nDocuments = 80
nTimepoints = 164
single_doc = False
preRun = False

# for raw timeseries classification
runTimeseries = False #whether to run classification on just the timeseries (no topic modeling)

# for phraseLDA
# Already run using topicalPhrases/run.sh but this get topic probabilities from that for classification
runPhraseLDA = False

# for word2vec model //UNFINISHED
runWord2Vec = False

# for bag of words classification
runBag = True
nGramsinCorpus = False #True
windowGrams = False
gramsOnly = False
mincount = 0 #30 #80 #150 #need massive number (like 3000) for network_wise words
# BEST: half_4letters + biGrams + 4 mincount + RF w/ 1000 estimators gives mean: 0.825 (vocab of 248 words)
###### without biGrams: (vocab of 236 words) gives around 0.8

# NEW BEST: 4wordwindow + biGrams + 150 mincount + SVM gives mean: 0.8625
    # this is with non-normalized windows so it probably not valid

# for doc2vec classification
runDoc2Vec = False

# for classification
nLabelOne = 90#70#90#40#70#30 #number of TDs
nLabelTwo = 90#70#90#40#70#30 #number of ADs
labels  = np.asarray([0] * nLabelOne + [1] * nLabelTwo)
nFolds = len(labels) #leave-one-out
nEstimators = 1000 #1000 #number of estimators for random forest classifier

runClassification = True # run classification on topic probabilities
#runClassification = False # OPTION ONLY FOR HDP ...run classification on document similarities

################################################
############## PROCESS DATA ####################
################################################

# Create dictionary with list of processed words for each document key
documents = getDocuments(scripts, delimiter, isCorpus, textNames)
#plot_Networkletters(documents, title, letters)

if network_wise:

    new_docs = {}
    for doc in documents:
        new_docs[doc] = []
        for word in documents[doc]:
            #documents[doc][documents[doc].index(word)] = 'DMN_' + word[0] + ' SN_' + word[1] + ' LECN_' + word[2] + ' RECN_' + word[3]
            new_docs[doc].append(' dmn_' + word[0])
            new_docs[doc].append(' sn_' + word[1])
            new_docs[doc].append(' lecn_' + word[2])
            new_docs[doc].append(' recn_' + word[3])

    documents = new_docs

if isCorpus and not runDoc2Vec:
    scripts = textNames[1:len(textNames)]

# Create dictionary with context lists for each word in each document
#contexts = getnGrams(scripts, nGrams, documents)

################################################
############### RUN MODELS #####################
################################################

topics = {}
topicProbs = {}
indivProbs = {}
svmACC = [0] * nModels
rfACC = [0] * nModels
kACC = [0] * nModels
enetACC = [0] * nModels
importances = [[]] * nModels
stds = [[]] * nModels
a = 0
for i in range(nModels):
   print "===================================================================================="
   print "Running Models for Iteration # %i on %s" %(i+1, path)
   print "====================================================================================\n"
   if runLDA:
       print "Topic Modeling (LDA)..\n"

       nWords = nWords + a
       topics[i], topicProbs[i], indivProbs[i]  = ldaModel(scripts,nTopics,nIters,nWords,documents) # run LDA to get topics
       a += 1
       data = np.asarray(indivProbs[i])

       ###############################
       # plotting mean probabilities #
       ###############################

       fig, ax = plt.subplots()
       index = np.arange(len(topicProbs[i]))
       ax.bar(index,topicProbs[i]) #s=20 should be abstracted to number of words in topics
       bar_width = 0.35
       plt.xticks(index + bar_width, (map(str,range(nTopics))))
       plt.show()

   elif runHDP:
       print "Topic Modeling (HDP)...\n"

       if mincount != 0:
           data, reducedDocuments, featureNames = bagOfWords(scripts, documents, nGramsinCorpus, mincount, windowGrams, gramsOnly)

           indivProbs[i] = hdpModel(scripts, reducedDocuments, tLimit, runClassification)
       else:
           indivProbs[i] = hdpModel(scripts, documents, tLimit, runClassification)
       data = np.asarray(indivProbs[i])

       print data

   elif runDTM:
       print "Topic Modeling (DTM)...\n"
       #this currently returns mean topic prob over time..probably useless
       data = DTModel(scripts, documents, nTopics, nDocuments, nTimepoints, single_doc, preRun, network_wise)

   elif runTimeseries:

       #### FOR RAW TIMESERIES ####
       timeseries = scipy.io.loadmat('texts/ADTD_timeseries.mat')
       #print len(timeseries['TDs'][0])
       #print (timeseries['ADs'][0])
       #data = timeseries['TDs'][0]#, timeseries['ADs'][0]))

       indiv = np.zeros((4, len(timeseries['ADs'][0][0][0])))
       data = np.asarray([indiv] * len(labels))
       for d in range(len(labels)):
           if d <= 39:
               data[d] = timeseries['TDs'][0][d]
           else:
               data[d] = timeseries['ADs'][0][d-40]

       data_size = len(data)
       data = data.reshape(data_size,-1)

   elif runPhraseLDA:
       indivProbs = f.getTopicProbs()
       data = np.asarray(indivProbs)

   elif runWord2Vec:
       data = word2vecModel(scripts, documents)

   elif runBag:
       data, newVocab, featureNames = bagOfWords(scripts, documents, nGramsinCorpus, mincount, windowGrams, gramsOnly)
       countFreq = np.sum(data, axis=0)
       #print countFreq, featureNames
       forBag = [scripts, documents, nGramsinCorpus, mincount]
       # need to run this in my LOOCV because using test doc in feature selection corpus

       #myfile = open('NYU_TD_4letter_half_BoW', 'wb')
       #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
       #wr.writerow(countFreq)
       #wr.writerow(featureNames)


   elif runDoc2Vec:
       data = doc2vecModel(scripts)

   else:
       #should be ACC: 1 for all classifiers
       print "Checking sanity...\n"
       data = np.asarray([[0, 0, 0, 0, 0]] * nLabelOne + [[1, 1, 1, 1, 1]] * nLabelTwo)

   print "Done.\n"

   ###### CLUSTERING #######

   #if not runTimeseries and not runBag:
    #   print "Clustering...\n"
     #  kACC[i] = kCluster(data, labels)
      # print "K_means ACC:", kACC[i]
       #print " "

   ###### CLASSIFICATION #######

   if not runDTM:
       print "Running Elastic Net...\n"
       enetACC[i] = eNetModel(data,labels,featureNames,scripts, documents, nFolds)
       #enetACC[i] = eNetModel(data, labels, nFolds, featureNames)
       print "eNet ACC:", enetACC[i], "\n"
       #
       #
       print "Running SVM...\n"
       svmACC[i] = svmModel(data,labels,featureNames,scripts, documents, nFolds)
       #svmACC[i] = svmModel(data, labels, nFolds, bagIt=forBag)
       print "svm ACC:", svmACC[i], "\n"

       print "Running RF with %i estimators...\n" %(nEstimators)
       rfACC[i], importances[i], stds[i] = rfModel(data,labels,featureNames,scripts, documents, nFolds, nEstimators)
       #rfACC[i], importances[i], stds[i] = rfModel(data, labels, nFolds, nEstimators, bagIt=forBag)
       idx = (-importances[i]).argsort()[:10]

       print "Top 10 features:"
       #for j in idx:
        #   print (featureNames[j], importances[i][j]), "std: ", stds[i][j]

       print "\nrf ACC:", rfACC[i], "\n"

if not runDTM:
    print "=================================="
    print "Mean Values for %i Models"%(i+1)
    print "==================================\n"
    #if not runTimeseries:
    #    print "kmeans acc mean:", np.mean(kACC)
    #print "enet acc mean:", np.mean(enetACC)
    #print "svm acc mean:", np.mean(svmACC)
    print "rf acc mean:", np.mean(rfACC)


###################################################################
# plotting word usage across topics w/ different number of words  #
###################################################################

#
# commonWords = [[]]
# for i in range(len(topics)):
#     if i != len(topics)-1:
#         max_index, max_value = max(enumerate(topicProbs[i]), key=operator.itemgetter(1))
#         max_index2, max_value2 = max(enumerate(topicProbs[i+1]), key=operator.itemgetter(1))
#         commonWords[0].append([word for word in topics[i][max_index] if word in set(topics[i+1][max_index2])])
# commonWords = (sum(sum(commonWords,[]),[]))
#
# wordcount={}
# for word in commonWords:
#     if word and word in wordcount:
#         wordcount[word] += 1
#     else:
#         wordcount[word] = 1
#
# sorted_wordcount = sorted(wordcount.items(), key=operator.itemgetter(1))
#
# fig, ax = plt.subplots()
# index = np.arange(len(sorted_wordcount))
# words = []
# counts = []
# for i in range(len(sorted_wordcount)):
#     words.append(sorted_wordcount[i][0])
#     counts.append(sorted_wordcount[i][1])
#
# ax.bar(index,counts) #s=20 should be abstracted to number of words in topics
# bar_width = 0.35
# plt.xticks(index + bar_width, words)
# plt.show()





#THIS STEP TAKES FAR TOO LONG
#Q = QbyContextinTopic(topics, contexts, scripts, nWords) # get co-occurrence matrix based on context words for each topic
#print Q

#sim = f.makeSim(Q) # make similarity matrix from Q

#f.mdsModel(sim, topics) # run MDS and plot results

#Q = QbyContextinDoc(documents, contexts, nGrams)
#print Q

#similarities = f.makeSim(topics,contexts,scripts,'byTopic')
#similarities = f.makeSim(documents,contexts,scripts,'general')

#f.nmfModelwAnchors(scripts, documents, nTopics)
#f.mdsModel(similarities, topics)

#print sim_df
#print contexts
#print topics
#for text in scripts:
#    print (contexts[text])

# wordCounts = f.wordCount(scripts)
#
# offset = 5
# topWords = {}
# topCounts = {}
# common = []
# for script in scripts:
#     counts ={}
#     words = [word[0] for word in wordCounts[script]]
#     counts = [count[1] for count in wordCounts[script]]
#     topWords[script] = words[len(words)-offset:len(words)]
#     topCounts[script] = counts[len(counts)-offset:len(counts)]
#
#     print script, topWords[script], topCounts[script]

# for script in scripts:
#     for sc in scripts:
#         if sc != script:
#             common.append(list(set(top50words[sc]).intersection(top50words[script])))
#
# #common = sum(common, [])
# print common
#f.bagOfWords('texts/biglebowski_script.txt')


#need to remove names of characters talking in script and also think of other noisy issues here
    # running list of some issues here...
    # dialects are used which would need to bee controlled for somehow
    #       i.e. more a that = more of that


#lebowski = f.wordCount('texts/biglebowski_script.txt')
#nihilism = f.wordCount('texts/nihilism_iep.txt')

#lebowskiWords = [x[0] for x in lebowski]
#nihilismWords = [x[0] for x in nihilism]

#offset = 300

#x = lebowskiWords[len(lebowskiWords)-offset:len(lebowskiWords)]
#y = nihilismWords[len(nihilismWords)-offset:len(nihilismWords)]

#print lebowski
#print len(lebowski), len(nihilism), list(set(x).intersection(y))
