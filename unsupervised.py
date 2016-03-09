'''
    Unsupervised learning functions
'''

from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities
from sklearn import cluster
from decimal import *
import numpy as np
import lda

def hdpModel(texts, documents, tLimit, forClass):
    def getKey(item):
        return item[1]

    textList = []
    for text in texts:
        textList.append(documents[text])

    if forClass:

        dictionary = corpora.Dictionary(textList)
        corpus = [dictionary.doc2bow(text) for text in textList]

        hdp = models.HdpModel(corpus, dictionary, T=tLimit,kappa=0.6)
        topicProbs = [[]] * len(texts)
        for text in range(len(texts)):
            topicProb = [0] * tLimit
            text_bow = dictionary.doc2bow(textList[text])
            #print hdp[text_bow]
            for prob in hdp[text_bow]:
                topicProb[prob[0]] = prob[1]
            topicProbs[text] = topicProb
        topics = hdp.print_topics(topics=151, topn=10)
        #print topics

        return topicProbs

    else:
        training_indices = range(10) + range(40,50)
        trainingList = [textList[x] for x in training_indices]
        testing_indices = range(10,40) + range(50,len(textList))
        testingList = [textList[x] for x in testing_indices]

        dictionary = corpora.Dictionary(trainingList)
        corpus = [dictionary.doc2bow(text) for text in trainingList]

        hdp = models.HdpModel(corpus, dictionary, T=tLimit)

        #transform corpus to hdp space and index it
        index = similarities.MatrixSimilarity(hdp[corpus])

        topicSims = [[]] * len(testingList)
        for text in range(len(testingList)):
            text_bow = dictionary.doc2bow(testingList[text])
            text_hdp = hdp[text_bow]
            sims = index[text_hdp]
            topicSims[text] = sims

        return topicSims



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

    for i in range(len(texts)):
        print("{} (top topic: {})".format(texts[i], doc_topic[i].argmax()))

    probs = np.array(doc_topic)
    meanProbs = np.mean(probs, axis=0)

    return topic_words, meanProbs, probs



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


def jsd(x,y): #Jensen-shannon divergence
    x = np.array(x)
    y = np.array(y)
    d1 = x*np.log2(2*x/(x+y))
    d2 = y*np.log2(2*y/(x+y))
    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d = 0.5*np.sum(d1+d2)
    return d


def kCluster(vectorized, labels):
    js_dist = np.zeros((len(vectorized), len(vectorized)))
    for t in range(len(vectorized)):
        for t2 in range(len(vectorized)):
            js_dist[t][t2] = jsd(vectorized[t], vectorized[t2])

    k_means = cluster.KMeans(n_clusters=2)

    k_means.fit(js_dist)

    acc = 0
    for l in range(len(labels)):
        if labels[l] == k_means.labels_[l]:
            acc += 1
    return Decimal(acc)/Decimal(len(labels))
