import os
import functions as f
#import matplotlib.pyplot as plt
import numpy as np
import operator
from processing import getDocuments
from contexts import getnGrams
from occurrences import *
from supervised import *
from unsupervised import *
import scipy

#path = 'texts/mPFC_ofMRI/'
path = 'texts/AD_TD_full_3letters/'

scripts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

# choose parameters
nModels = 10

delimiter = 'none' #or ',''
nTopics = 10
nWords = 3 #is actually n - 1 (so 3 for 2 words)
nGrams = 10
nIters = 500

tLimit = 10

nLabelOne = 40 #30
nLabelTwo = 40 #30 for class on doc sims
labels  = np.asarray([0] * nLabelOne + [1] * nLabelTwo)

runClassification = True
#runClassification = False # <--- plot the similarities based on whether they were most similar to TD or AD

documents = getDocuments(scripts, delimiter) # get dict with list of processed word/document

contexts = getnGrams(scripts, nGrams, documents) # get dict with dict of context lists for each word in each document

topics = {}
topicProbs = {}
indivProbs = {}
svmACC = [0] * nModels
rfACC = [0] * nModels
kACC = [0] * nModels
a = 0
for i in range(nModels):
   print "=================================="
   print "Running Model # %i" %(i+1)
   print "=================================="

   nWords = nWords + a
   #topics[i], topicProbs[i], indivProbs[i]  = ldaModel(scripts,nTopics,nIters,nWords,documents) # run LDA to get topics
   print "Topic Modeling..."
   print " "
   indivProbs[i] = hdpModel(scripts, documents, tLimit, runClassification)
   a += 1

   data = np.asarray(indivProbs[i])
   #data = np.asarray([[0, 0, 0]] * nLabelOne + [[1, 1, 1]] * nLabelTwo) # <---- sanity check data set (should be ACC: 1)

   ###### CLUSTERING #######
   print "Clustering..."
   kACC[i] = kCluster(data, labels)
   print "K_means ACC:", kACC[i]
   print " "
   ###### CLASSIFICATION #######

   # timeseries = scipy.io.loadmat('texts/ADTD_timeseries.mat')
   # print len(timeseries['TDs'][0][0][3])

   print "Running SVM..."
   svmACC[i] = svmModel(data, labels)
   print "svm ACC:", svmACC[i]
   print " "

   print "Running RF..."
   rfACC[i] = rfModel(data, labels)
   print "rf ACC:", rfACC[i]
   print " "

print "=================================="
print "Mean Values for %i Models"%(i+1)
print "=================================="
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
