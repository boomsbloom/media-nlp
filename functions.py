import string
import operator
import numpy as np
import pandas as pd
import math
from decimal import *
from scipy import stats, integrate, sparse
import scipy.spatial.distance
#import matplotlib.pyplot as plt
#import seaborn as sns
import lda
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, manifold, svm, grid_search, cluster
from sklearn.ensemble import RandomForestClassifier

def docCluster(topicProbs, label):
        dist = scipy.spatial.distance.pdist(topicProbs, 'euclidean')
        dist_mat = scipy.spatial.distance.squareform(dist)
        print dist_mat

def makeSim(Q):
    '''
        Currently operates by computing number of shared context words

        This function receives a normalized co_occurrence matrix based on the
        number of shared context words as computed by getnGrams/getKWIC for
        each topic and then returns a euclidian distance similarity matrix

        number of shared context words <--- within document and between
        number of documents in common for each topic word <--- between docs

        Should extend to also compute the number of documents in common for each
        topic word (this would be the number of documents that the words are
        found together in)
    '''

    sim_mat = {}
    for q in range(len(Q)):
        sim = scipy.spatial.distance.pdist(Q[q], 'euclidean')
        sim_mat[q] = scipy.spatial.distance.squareform(sim)

    return sim_mat

def QbyKWIC(wordLists, contexts, texts): #figure out whats going on here now...
    topics = wordLists                  #topics is 5 19 word arrays
    Q = {}                              # contexts is 5 lists with contexts for each word
    for t in range(len(contexts)):
        Q[t] = np.zeros((len(topics[t]), len(topics[t])))
        for w in range(len(topics[t])):
            for w2 in range(len(contexts[t])):
                for con in contexts[t][w2]:
                    for w3 in range(len(con)):
            #commonWords = list(set(contexts[t][topics[t][w]][0][w]).intersection(contexts[t][topics[t][w2]][0][w]))
            #            if w3 != int(len(con)/2 + 0.5) and :
                            print topics[t][w], w3
            try:
                for w in range(len(contexts[t][topics[t][w3]][0])):
                    for con in contexts[t][topics[t][w3]][0][w]:
                        if con != topics[t][w3]:
                            for w2 in range(len(topics[t])):
                                commonWords = list(set(contexts[t][topics[t][w3]][0][w]).intersection(contexts[t][topics[t][w2]][0][w]))
                                #co-occurrence equals the number of shared content words / total number of co-occurrences in order to normalize
                                Q[t][w3,w2] = Decimal(len(commonWords))/Decimal((len(topics[t]) * (len(topics[t])+1)) / 2)
            except:
                pass
    return Q

def mdsModel(sims, topics):
    seed = np.random.RandomState(seed=3)
    for t in range(len(topics)):

        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(sims[t]).embedding_
        #rescale data
        npos = mds.fit_transform(sims[t], init=pos)
        fig, ax = plt.subplots()
        ax.scatter(npos[:, 0], npos[:, 1], s=len(topics[t]), c='b') #s=20 should be abstracted to number of words in topics

        for i, txt in enumerate(topics[t]):
            ax.annotate(txt, (npos[:, 0][i],npos[:, 1][i]))

        plt.show()


def sortContext(contexts, documents, texts, topics):
    topicContexts = {}
    for d in range(len(documents)):
        docContexts = []
        for word in contexts[texts[d]]:
            for t in range(len(topics)):
                for word2 in topics[t]:
                    if word == word2: # THIS MAY BE GRABBING ONLY FIRST INSTANCE..DOUBLE CHECK
                        docContexts.append(contexts[texts[d]][word])
                topicContexts[t] = docContexts
    return topicContexts


def getVocab(texts, docs):
    documents = []
    for doc in docs:
        documents.append((' ').join(docs[doc]))
        #docs[doc].append((' ').join(docs[doc]))
    #print documents
    vectorizer = CountVectorizer(stop_words = None,
                                 tokenizer = None,
                                 token_pattern =  '(?u)\\b\\w+\\b' , #use this to keep single characters
                                 preprocessor = None)
    dtm = vectorizer.fit_transform(documents).toarray()

    vocab = np.array(vectorizer.get_feature_names())

    return vocab, dtm


def nmfModelwAnchors(texts, docs, nTopics):
    # get co_occurrence matrix
    Q = QbyDocument(texts, docs)
    print Q
    # row normalize Co_occurrence matrix


    # Qn = {}
    # for t in range(nTopics):
    #     s = Q[t].sum(axis=1)
    #     Qn[t] = Q[t] / s[:, np.newaxis]
    #
    # #get anchor word permuatation indices and diagonal of R
    # (p, r) = getAnchors(Qn, nTopics)
    #
    # A = {}
    # for t in range(nTopics):
    #     #compute intensity matrix for each word
    #     A[t] = getA(Qn[t], s, p[t]) #s is of size number words


def QbyDocument(texts, docs):
    for d in docs:
        print d[0]


    # vocab, dtm = getVocab(texts, docs)
    # nDocs = float(dtm.shape[0])
    # nVocab = float(dtm.shape[1])
    #
    # dtm = sparse.csc_matrix(dtm)
    #
    # diag_dtm = np.zeros(nVocab)
    #
    # for j in xrange(dtm.indptr.size - 1):
    #     start = dtm.indptr[j]
    #     end = dtm.indptr[j + 1]
    #
    #     wpd = float(np.sum(dtm.data[start:end]))
    #     row_indices = dtm.indices[start:end]
    #
    #     diag_dtm[row_indices] += dtm.data[start:end]/(wpd*(wpd-1))
    #     dtm.data[start:end] = dtm.data[start:end]/math.sqrt(wpd*(wpd-1))
    #
    # Q = dtm*dtm.transpose()/nDocs
    # print Q
    # Q = Q.todense()
    # Q = np.array(Q, copy=False)
    # diag_dtm = diag_dtm/nDocs
    # Q = Q - np.diag(diag_dtm)

    return Q

def getTopicProbs():
    g = open('topicalPhrases/output/outputFiles/topics.txt','r')

    probabilities = []
    for topics in g:
    	l = topics.split(',')
    	length = Decimal(len(l) - 1) # -1 to account for linebreak index
        length = float(length)
    	topicProb = dict((i, l.count(i)/length) for i in l)
    	topicProb = topicProb.values()[1:len(topicProb.values())]
    	probabilities.append(topicProb)

    return probabilities

def getA(Qn, s, p):
    Tt = Qn[p,:]


def getAnchors(Qn, nTopics):
    '''
        Use dense pivoted QR factorization to select anchor words.
        Anchor words permutation indices and diagonal of R returned as outputs.
    '''
    q = {}
    r = {}
    p = {}
    for t in range(nTopics):
        #Qn.shape
        q[t], r[t], p[t] = scipy.linalg.qr(Qn[t],pivoting=True)
    return p, r


def wordCount(texts):
    sorted_wordcount = {}
    for text in texts:
        script = open(text, 'r')
        wordcount={}
        for word in script.read().split():
            word = preProcess(word)
            if word and word in wordcount:
                wordcount[word] += 1
            else:
                wordcount[word] = 1

        sorted_wordcount[text] = sorted(wordcount.items(), key=operator.itemgetter(1))

    return sorted_wordcount















#
