import string
import operator
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import lda
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition


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

def getVocab(texts):
    words = []
    documents = []
    for text in texts:
        if text != "texts/.DS_Store":
            script = open(text, 'r')
            print "Preprocessing words in %s...\n"%(text)
            for word in script.read().split():
                word = preProcess(word) #processing each word
                if word:
                    words.append(word)
        documents.append((' ').join(words))
    vectorizer = CountVectorizer(stop_words = None,
                                 tokenizer = None,
                                 preprocessor = None) #only words with at least 20 usages..might be too high
    dtm = vectorizer.fit_transform(documents).toarray()

    vocab = np.array(vectorizer.get_feature_names())
    return vocab, dtm

def ldaModel(texts):
    vocab, dtm = getVocab(texts)
    model = lda.LDA(n_topics=3, n_iter=500, random_state=1)
    model.fit(dtm)
    topic_word = model.topic_word_
    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    doc_topic = model.doc_topic_
    for i in range(len(texts)):
        print("{} (top topic: {})".format(texts[i], doc_topic[i].argmax()))


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
