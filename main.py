import os
import functions as f
from processing import getDocuments
from contexts import getnGrams
from occurrences import *

path = 'texts/mPFC_ofMRI/'

scripts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

#scripts = ['texts/coen/aseriousman_script.txt','texts/coen/burnafterreading_script.txt',
#            'texts/coen/truegrit_script.txt','texts/coen/biglebowski_script.txt',
#            'texts/coen/obrother_script.txt']
#scripts = [scripts[3]]
#scripts = ['texts/inglouriousbasterds_script.txt']
#scripts = ['atrophy.txt']
#scripts = ['atrophy.txt','bear.txt','epilogue.txt','kettering.txt',
#            'prologue.txt','shiva.txt','sylvia.txt','thirteen.txt',
#            'two.txt','wake.txt']

#scripts = sorted([os.path.join(fn) for fn in os.listdir(path)])
#scripts = [path + script for script in scripts]

#scripts = ['RatA_ON.txt','RatB_ON.txt','RatC_ON.txt','RatD_ON.txt']
          #'RatA_OFF.txt','RatB_OFF.txt','RatC_OFF.txt','RatD_OFF.txt']

#scripts = sorted([os.path.join('texts/antlers_hospice/',song) for song in scripts])

#f.nmfModel(scripts)
#f.ldaModel(scripts,3,500) #3,500

# choose parameters
delimiter = ',' #or 'none'
nTopics = 2
nWords = 8
nGrams = 5
nIters = 500

#SR suggested vector quantization instead of SAX (so you just cluster, choose the top one, then make it your label)

documents = getDocuments(scripts, delimiter) # get dict with list of processed word/document

contexts = getnGrams(scripts, nGrams, documents) # get dict with dict of context lists for each word in each document

topics = f.ldaModel(scripts,nTopics,nIters,nWords,documents) # run LDA to get topics

# THIS STEP TAKES FAR TOO LONG
Q = QbyContextinTopic(topics, contexts, scripts, nWords) # get co-occurrence matrix based on context words for each topic
print Q

sim = f.makeSim(Q) # make similarity matrix from Q

f.mdsModel(sim, topics) # run MDS and plot results

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
