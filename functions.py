import string
import operator
import numpy as np
import pandas as pd
import math
from decimal import *
from scipy import stats, integrate, sparse
import scipy.spatial.distance
import matplotlib.pyplot as plt
import seaborn as sns
import lda
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, manifold

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


def ldaModel(texts,topics,iters, nWords, documents):
    vocab, dtm = getVocab(texts, documents)
    model = lda.LDA(n_topics=topics, n_iter=iters, random_state=1)
    model.fit(dtm)
    topic_word = model.topic_word_
    n_top_words = nWords
    topic_words = {}
    for i, topic_dist in enumerate(topic_word):
        topic_words[i] = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words[i])))
    doc_topic = model.doc_topic_
    #for i in range(len(texts)):
    #    print("{} (top topic: {})".format(texts[i], doc_topic[i].argmax()))
    return topic_words


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

def bagOfWords(text):
    script = open(text, 'r')
    print "Preprocessing words...\n"
    words = []
    for word in script.read().split():
        word = preProcess(word) #processing each word
        if word:
            words.append(word)

    print "Creating bag of words model...\n"
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = 5000)

    #fit model and tranform to feature vectors
    train_data_features = vectorizer.fit_transform(words)

    train_data_features = train_data_features.toarray() #convert to numpy array

    #shape of feature set
    #print train_data_features.shape

    #get our vocabulary
    vocab = vectorizer.get_feature_names()
    # print vocab

    # Sum count of each word
    sums = np.sum(train_data_features, axis=0)

    # will need to sort of whatever for this to be useful
    # sns.distplot(sums[range(100)])
    # sns.axlabel(vocab[range(100)])
    # plt.show()

    # For each, print the vocabulary word and the number of times it appears
    #for tag, count in zip(vocab, dist):
    #    print count, tag












#
