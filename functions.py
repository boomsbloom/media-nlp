import string
import operator
import numpy as np
import pandas as pd
from scipy import stats, integrate
import scipy.spatial.distance
import matplotlib.pyplot as plt
import seaborn as sns
import lda
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, manifold


def preProcess(word):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    punctuation = set(string.punctuation)
    word = word.lower() # make all words lowercase
    word = ''.join(char for char in word if char not in punctuation) #remove punctuation
    word = unicode(word, errors='ignore')
    #print word
    #word = lemmatizer.lemmatize(word) #run lemmatizer and stemmer (questionable performance)
    #word = stemmer.stem(word)
    if word not in stopwords.words("english"): #remove stopwords
        return word
    else:
        return False


def getDocuments(texts):
    words = []
    documents = []
    for text in texts:
        print "Processing words in %s..." %(text) #preprocessing twice right now which i dont need to do...
        if text != "texts/gutenberg/.DS_Store":
            script = open(text, 'r')
            for word in script.read().split():
                word = preProcess(word)
                if word:
                    words.append(word)
        documents.append(words)
    return documents


def getKWIC(texts, ngrams, n):
    kwicdict = {}
    for text in texts:
        keyindex = len(ngrams[text][0]) // int(n/2 + 0.5) #generalize this! 5 word context isn't enough i think
        dic = {}
        for k in ngrams[text]:
            if k[keyindex] not in dic:
                dic[k[keyindex]] = [k]
            else:
                dic[k[keyindex]].append(k)
        kwicdict[text] = dic
    return kwicdict


def getnGrams(texts, n, topics, documents, option):
    ngrams = {}
    for j in range(len(documents)):
        ngrams[texts[j]] = ([documents[j][i:i+n] for i in range(len(documents[j])-(n-1))])
        # this is imperfect because it doesn't count context for the last n-1 entries
    if option == 'byTopic':
        return sortContext(getKWIC(texts, ngrams, n), documents, texts, topics)
    else:
        return getKWIC(texts, ngrams, n)


def makeSim(wordLists, contexts, texts, option): #this makes similarity matrices for each topic
                               # number of shared context words <--- within document and between

                               # number of documents in common for each topic word <--- between docs
    topics = wordLists
    co_occurrence = {}
    sim_mat = {}
    if option == 'byTopic':
        for t in range(len(contexts)):
            co_occurrence[t] = np.zeros((len(topics[t]), len(topics[t])))
            for w3 in range(len(topics[t])):
                try:
                    for w in range(len(contexts[t][topics[t][w3]][0])):
                        for con in contexts[t][topics[t][w3]][0][w]:
                            if con != topics[t][w3]:
                                for w2 in range(len(topics[t])):
                                    commonWords = list(set(contexts[t][topics[t][w3]][0][w]).intersection(contexts[t][topics[t][w2]][0][w]))
                                    co_occurrence[t][w3,w2] = len(commonWords) #number of shared content words
                                    #if con == topics[t][w2]:
                                    #    co_occurrence[t][w3,w2] = 1 #binary; whether topic word co-occurs with other in n-gram context
                except:
                    pass
            #print co_occurrence[t]
            sim = scipy.spatial.distance.pdist(co_occurrence[t], 'euclidean')
            sim_mat[t] = scipy.spatial.distance.squareform(sim)

    else:

        # get number of shared context words for whole document

        # for t in range(len(contexts)):
        #
        #     co_occurrence[t] = np.zeros((len(topics[t]), len(topics[t])))
        #     for w1 in range(list(wordLists[t])):
        #         for w2 in range(list(wordLists[t])):
        #             try:
        #                 print contexts[texts[t]][wordLists[t][w1]]
        #             except:
        #                 pass
            #    for context in contexts[texts[t]]:
            #        print len(contexts[texts[t]])
                    #print t, word, context
            #print len(wordLists[t])
            #print len(contexts[texts[t]])
            #try:


    #sim_df = pd.DataFrame(sim_mat[0], index=topics[0], columns=topics[0])
    #print sim_df
    #print co_occurrence[0]
    #plt.matshow(sim_df)
    #plt.show()
    return sim_mat


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

        #sims[t] = sims[t].max() / sims[t] * 100
        #sims[t][np.isinf(sims[t])] = 0

        plt.show()


def sortContext(contexts, documents, texts, topics):
    contextPerTopic = {}
    for text in texts:
        for t in range(len(topics)):
            contextPerWord = {}
            for word in topics[t]:
                con = []
                for cWord in contexts[text]:
                    if word == cWord:
                        con.append(contexts[text][cWord])
                contextPerWord[word] = con
            contextPerTopic[t] = contextPerWord
    return contextPerTopic


def getVocab(texts, docs):
    documents = []
    for doc in docs:
        documents.append((' ').join(doc))
    vectorizer = CountVectorizer(stop_words = None,
                                 tokenizer = None,
                                 preprocessor = None) #only words with at least 20 usages..might be too high
    dtm = vectorizer.fit_transform(documents).toarray()

    vocab = np.array(vectorizer.get_feature_names())
    return vocab, dtm

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
    return topic_words


def nmfModel(texts):
    vocab, dtm = getVocab(texts)

    num_topics = 20
    num_top_words = 20
    clf = decomposition.NMF(n_components=num_topics, random_state=1)

    print "Fitting NMF model...\n"

    doctopic = clf.fit_transform(dtm)

    topic_words = []
    for topic in clf.components_:
        word_index = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append([vocab[i] for i in word_index])

    #scaling so that component values for each document sum to 1
    doctopic = doctopic/np.sum(doctopic, axis=1, keepdims=True)

    text_names = np.asarray(texts)

    doctopic_original = doctopic.copy()

    num_texts = len(set(text_names))

    doctopic_grouped = np.zeros((num_texts, num_topics))

    for i, text in enumerate(sorted(set(text_names))):
        doctopic_grouped[i,:] = np.mean(doctopic[text_names == text, :], axis=0)

    doctopic = doctopic_grouped

    texts = sorted(set(text_names))

    print "Top NMF topics in..."

    for i in range(len(doctopic)):
        top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
        top_topics_str = ' '.join(str(t) for t in top_topics)
        print "{}: {}".format(texts[i], top_topics_str)

    for t in range(len(topic_words)):
        print "Topic {}: {}".format(t, ' '.join(topic_words[t][:15]))

    # indices = []
    # for index in enumerate(sorted(set(text_names))):
    #     indices.append(index)
    #
    # avg = np.mean(doctopic[indices, :], axis=0)
    #
    # rank = np.argsort(avg)[::-1]
    #
    # print rank

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
