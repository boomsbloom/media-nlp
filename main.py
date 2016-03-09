import os, operator, scipy
import functions as f
#import matplotlib.pyplot as plt
import numpy as np
from processing import getDocuments
from contexts import getnGrams
from occurrences import *
from supervised import *
from unsupervised import *

#################################################
################ LOAD INPUT DATA ################
#################################################

path = 'texts/AD_TD_full_4letters/'
textNames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

# choose whether input is one document with your whole corpus in it (to be split up)
# ..or whether input is multiple documents
isCorpus = True
#isCorpus = False

if isCorpus:
    scripts = 'texts/full_4letter_phrase_corpus.txt'
else:
    scripts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
    if len(scripts) > 80: #removing .DS_Store
       scripts = scripts[1:len(scripts)]

#################################################
############### CHOOSE PARAMETERS ###############
#################################################

nModels = 10 # number of times you want modeling to run
nGrams = 10 # number of words in context ..only if running context calculation

# for LDA
runLDA = False # whether to run LDA
delimiter = 'none' #or ',' type of delimiter between your words in the document
nTopics = 10 # number of topics to create
nWords = 3 # number of words per topic; is actually n - 1 (so 3 for 2 words)
nIters = 500 # number of iterations for sampling

# for HDP
runHDP = True # whether to run HDP
tLimit = 150 # limit on number of topics to look for (default is 150)
# note: larger the limit on topics the more sparse the classification matrix

# for raw timeseries classification
runTimeseries = False #whether to run classification on just the timeseries (no topic modeling)

# for classification
nLabelOne = 40 #number of TDs
nLabelTwo = 40 #number of ADs
labels  = np.asarray([0] * nLabelOne + [1] * nLabelTwo)
nFolds = len(labels) #leave-one-out

runClassification = True # run classification on topic probabilities
#runClassification = False # OPTION ONLY FOR HDP ...run classification on document similarities

################################################
############## PROCESS DATA ####################
################################################

# Create dictionary with list of processed words for each document key
documents = getDocuments(scripts, delimiter, isCorpus, textNames)

if isCorpus:
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
a = 0
for i in range(nModels):
   print "=================================="
   print "Running Models for Iteration # %i" %(i+1)
   print "==================================\n"
   print "Topic Modeling...\n"
   if runLDA:

       nWords = nWords + a
       topics[i], topicProbs[i], indivProbs[i]  = ldaModel(scripts,nTopics,nIters,nWords,documents) # run LDA to get topics
       a += 1
       data = np.asarray(indivProbs[i])

   elif runHDP:

       indivProbs[i] = hdpModel(scripts, documents, tLimit, runClassification)
       data = np.asarray(indivProbs[i])

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

   else:
       #sanity check data set (should be ACC: 1)
       data = np.asarray([[0, 0, 0]] * nLabelOne + [[1, 1, 1]] * nLabelTwo)


   print "Done.\n"

   ###### CLUSTERING #######

   if not runTimeseries:
       print "Clustering...\n"
       kACC[i] = kCluster(data, labels)
       print "K_means ACC:", kACC[i]
       print " "

   ###### CLASSIFICATION #######

   print "Running SVM...\n"
   svmACC[i] = svmModel(data, labels, nFolds)
   print "svm ACC:", svmACC[i], "\n"

   print "Running RF...\n"
   rfACC[i] = rfModel(data, labels, nFolds)
   print "rf ACC:", rfACC[i], "\n"

print "=================================="
print "Mean Values for %i Models"%(i+1)
print "==================================\n"
if not runTimeseries:
    print "kmeans acc mean:", np.mean(kACC)
print "svm acc mean:", np.mean(svmACC)
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

###############################
# plotting mean probabilities #
###############################

# fig, ax = plt.subplots()
# index = np.arange(len(topicProbs))
# ax.bar(index,topicProbs) #s=20 should be abstracted to number of words in topics
# bar_width = 0.35
# plt.xticks(index + bar_width, (map(str,range(nTopics))))
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
