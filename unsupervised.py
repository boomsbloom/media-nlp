'''
    Unsupervised learning functions
'''

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from gensim import corpora, models, similarities, utils
#from gensim.models.doc2vec import LabeledSentence
#from gensim.models import Doc2Vec
#import gensim.models.wrappers as wrappers
from random import shuffle
from sklearn import cluster
from decimal import *
import numpy as np
import pandas as pd
import lda, nltk, time

def getKey(item):
    return item[1]

def DTModel(texts, documents, nTopics, nDocuments, nTimepoints, single_doc, preRun, network_wise):
    if network_wise:
        sent_length = 16
    else:
        sent_length = 4
    if not single_doc:
        textList = []
        #windowList = {}
        windowList = []
        #texts = texts[0:1]
        for text in texts:
            sentence = []
            sent_end = sent_length
            #windowList[text] = []
            for w in range(len(documents[text])):
                if w == sent_end:
                    sent_end+=sent_length
                    #windowList[text].append(sentence)
                    windowList.append(sentence)
                    sentence = []
                    sentence.append(documents[text][w])
                else:
                    sentence.append(documents[text][w])
            windowList.append(sentence)

        start_time = time.clock()
        dictionary = corpora.Dictionary(windowList)
        corpus = [dictionary.doc2bow(sen) for sen in windowList]
        timeslices = [len(windowList)/nTimepoints] *  nTimepoints
        print "%s seconds elapsed for corpus generation\n" %(time.clock() - start_time)


        if preRun:
            #modelFile = 'fitted_dtm_%itopic_AD' %(nTopics)
            modelFile = 'fitted_dtm_%itopic_network-wise' %(nTopics)
            dtm = wrappers.DtmModel.load(modelFile)
        else:
            start_time = time.clock()
            dtm = wrappers.DtmModel('dtm-master/bin/dtm-darwin64',corpus,timeslices,num_topics=nTopics,id2word=dictionary,initialize_lda=True)
            #modelfile = 'fitted_dtm_%itopic_AD' %(nTopics)
            modelfile = 'fitted_dtm_%itopic_TD_network-wise' %(nTopics)
            dtm.save(modelfile)
            print "%s seconds elapsed for model fitting\n" %(time.clock() - start_time)

        gammas = dtm.gamma_ #topic proportion for each document at each t

        gammas3d = np.reshape(gammas,(nTimepoints,nDocuments,nTopics))

        gamma_mu = np.zeros((nDocuments, nTopics))
        for s in range(nDocuments):
            for j in range(nTopics):
                gamma_mu[s,j] = np.mean([gammas3d[t,s,j] for t in range(nTimepoints)], axis = 0)

        #print len(loaded.show_topics(topics=-1,times=1,topn=1))
        #print np.fromfile(loaded.fout_prob(),dtype=float)

        dynamic_topics = {}
        for top in range(nTopics):
            dynamic_topics[top] = {}
            for ts in range(nTimepoints):
                dynamic_topics[top][ts] = {}
                tProbs = dtm.show_topic(top, ts, topn=10) #10
                for word in tProbs:
                    dynamic_topics[top][ts][word[1]] = word[0]
                #print "topic:", top, "slice:", ts
                #dynamic_topics[top][ts] = pd.DataFrame.from_dict(dynamic_topics[top][ts],orient='index')

            s = []
            for ts in range(nTimepoints): #get unique words in topic
                dic = dynamic_topics[top][ts]
                key_list = list(set(key for key in dic.keys()))
                s = s + key_list
                s = list(set(word for word in s))

            df = pd.DataFrame(index=s, columns=range(nTimepoints))
            df = df.fillna(0)

            for ts in range(nTimepoints):
                dic = dynamic_topics[top][ts]
                for key in dic.keys():
                    df.loc[key,ts] = dic.get(key)

            #filename = 'dynamic_topics/%i_topics_AD/dynamic_data_topic_%i_1word'%(nTopics,top)
            filename = 'dynamic_topics/%i_topics_TD_network-wise/dynamic_data_topic_%i'%(nTopics,top)
            df.to_csv(filename)
            #filename2 = 'dynamic_topics/%i_topics_AD/gammas.csv'%(nTopics)
            filename2 = 'dynamic_topics/%i_topics_TD_network-wise/gammas.csv'%(nTopics)
            np.savetxt(filename2,gammas,delimiter=',')
        print 'Dynamics saved for all topics.\n'

    return gamma_mu

# def word2vecModel(texts, documents):
#     textList = []
#     for text in texts:
#         textList.append(documents[text])
#     print textList

def doc2vecModel(texts):

    class LabeledLineSentence(object):
        def __init__(self, sources):
            self.sources = sources

            flipped = {}

            # make sure that keys are unique
            for key, value in sources.items():
                if value not in flipped:
                    flipped[value] = [key]
                else:
                    raise Exception('Non-unique prefix encountered')

        def __iter__(self):
            for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

        def to_array(self):
            self.sentences = []
            for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
            return self.sentences

        def sentences_perm(self):
            shuffle(self.sentences)
            return self.sentences

    sources = {texts:'Subject'}
    sentences = LabeledLineSentence(sources)

    nFeatures = 400
    model = Doc2Vec(min_count=1, window=7, size=nFeatures, sample=1e-4, negative=5, workers=8)
    model.build_vocab(sentences.to_array())
    for epoch in range(10):
        model.train(sentences.sentences_perm())

    feature_arrays = np.zeros((len(sentences.to_array()), nFeatures))
    for i in range(len(sentences.to_array())):
        sentence_label = 'Subject_' + str(i)
        feature_arrays[i] = model.docvecs[sentence_label]

    return feature_arrays


def bagOfWords(texts, documents, nGram, toReduce, windowGrams, gramsOnly):
   textList = []
   for text in texts:
       if windowGrams:
           begin = 0
           end = 4
           bigramList = []
           for s in range(len(documents[text])/4):
               sentence = documents[text][begin:end]
               for item in nltk.bigrams(sentence):
                   bigramList.append('_'.join(item))
               begin += 4
               end += 4
           textList.append(' '.join(bigramList))
       elif gramsOnly:
          bigramList = []
          for item in nltk.bigrams(" ".join(documents[text]).split()):
              bigramList.append('_'.join(item))
          textList.append(' '.join(bigramList))
       elif not nGram:
           textList.append(" ".join(documents[text]))
       else:
           bigramList = []
           for item in nltk.bigrams(" ".join(documents[text]).split()):
               bigramList.append('_'.join(item))
           tempList = bigramList + documents[text]
           textList.append(' '.join(tempList))

   print "Creating bag of words model...\n"

   vectorizer = CountVectorizer(analyzer = "word", #Count or Tfidf Vectorizer
                                tokenizer = None,
                                preprocessor = None,
                                stop_words = None,
                                ngram_range= (1, 1),
                                max_features = None)


   #fit model and tranform to feature vectors
   tdf = vectorizer.fit_transform(textList)

   train_data_features = tdf.toarray() #convert to numpy array

   featureNames = vectorizer.get_feature_names()

   reducedTextDic = []

   if toReduce != 0:

       vocab = vectorizer.get_feature_names()

       dist = np.sum(train_data_features, axis=0)

       # Removing words with a count less than toReduce
       newVocab = []
       for count in range(len(dist)):
           if dist[count] > toReduce:
               newVocab.append(vocab[count])

       reducedTextList = []
       reducedTextDic = {}
       if nGram:
           print "including bigrams in vocabulary\n"

       for text in texts:
           if not nGram:
               reducedTextList.append(" ".join([i for i in documents[text] if i in set(newVocab)]))
               reducedTextDic[text] = ([i for i in documents[text] if i in set(newVocab)])
           else:
               bigramList = []
               for item in nltk.bigrams(" ".join(documents[text]).split()):
                   bigramList.append('_'.join(item))
               tempList = bigramList + documents[text]
               reducedTextList.append(" ".join([i for i in tempList if i in set(newVocab)]))
               reducedTextDic[text] = ([i for i in tempList if i in set(newVocab)])

       vectorizer = CountVectorizer(analyzer = "word", #Count or Tfidf Vectorizer
                                    tokenizer = None,
                                    preprocessor = None,
                                    stop_words = None,
                                    ngram_range= (1, 1),
                                    max_features = None)
       #fit model and tranform to feature vectors
       tdf = vectorizer.fit_transform(reducedTextList)

       featureNames = vectorizer.get_feature_names()

       train_data_features = tdf.toarray() #convert to numpy array

   print "number of words in vocabulary:", len(train_data_features[0]), "\n"
#   normed_data_features = []
#   for feature_list in train_data_features:
#       normed_feats = [float(Decimal(num)/Decimal(sum(feature_list))) for num in feature_list]
#       normed_data_features.append(normed_feats)

#   train_data_features = np.array(normed_data_features)
   return train_data_features, reducedTextDic, featureNames


def hdpModel(texts, documents, tLimit, forClass):

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
            print text_hdp
            print index
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
