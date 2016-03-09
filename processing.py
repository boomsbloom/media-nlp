from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

'''
functions to process documents. Performs preprocessing and then returns document
as dict: documents[document] = list of words
'''

def preProcess(word):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    punctuation = set(string.punctuation)
    word = word.lower() # make all words lowercase
    word = ''.join(char for char in word if char not in punctuation) #remove punctuation
    word = word.rstrip()
    word = unicode(word, errors='ignore')
    #word = lemmatizer.lemmatize(word) #run lemmatizer and stemmer (questionable performance)
    #word = stemmer.stem(word)
    if word not in stopwords.words("english"): #remove stopwords
        return word
    else:
        return False

def getDocuments(texts, delimiter, corpus, textNames):
    documents = {}
    if not corpus:
        for text in texts:
            index = 0
            words = []
            #print "Processing words in %s..." %(text)
            script = open(text, 'r')
            if delimiter == ',':
                for word in script.read().split(','):
                    index += 1
                    word = preProcess(word)
                    if word:
                        words.append(word)
            else:
                for word in script.read().split():
                    index += 1
                    word = preProcess(word)
                    if word:
                        words.append(word)

            if text != 'texts/AD_TD_full_4letters/.DS_Store':
                documents[text] = words

    else:
        script = open(texts, 'r')
        index = 0
        for doc in script.read().split('\n'):
            words = []
            for word in doc.split(','):
                words.append(word)
            if textNames[index] != 'texts/AD_TD_full_4letters/.DS_Store':
                documents[textNames[index]] = words
            index += 1

    return documents
