import numpy as np
import pandas as pd
from decimal import *


'''
this will be where all occurrence counting functions go
so like bag of words, co-occurrence calculation, etc
'''

# normalizing portion of this function is questionable right now
def QbyContextinDoc(documents, contexts):
    # construct a co-occurrence matrix for each document
    Q = {}
    for doc in documents:
        #print set(documents[doc])
        unique_words = list(set(documents[doc]))
        Q_np = np.zeros((len(unique_words), len(unique_words))) #need to make Q only of unique words in document
        Q[doc] = pd.DataFrame(Q_np,columns=unique_words,index=unique_words)
        for w in range(len(documents[doc])):
            word = documents[doc][w]
            try: # this is necessary because not all words have a context right now (see contexts.py) and should be fixed
                for context in contexts[doc][word]:
                    for w2 in range(len(context)):
                        context_word = context[w2]
                        if word != context_word: #removing diagonals
                            Q[doc][word][context_word] += 1
            except:
                pass

        # normalize it
        for row in unique_words:
            for col in unique_words:
                Q[doc][row][col] = Decimal(Q[doc][row][col])/Decimal((len(unique_words) * (len(unique_words)+1))/ 2)
    return Q

# This is embarassingly inefficient. Works for now but need to get this moving in a more pythonic way
def QbyContextinTopic(wordLists, contexts, texts, n):
    topics = wordLists
    Q = {}
    currInd = 0
    for topic in topics:
        Q_np = np.zeros((n-1, n-1))
        Q[currInd] = pd.DataFrame(Q_np,columns=topics[topic],index=topics[topic])
        for context in contexts:
            for word in topics[topic]:
                for word2 in topics[topic]:
                    try: #because not all have contexts right now...which needs to be fixed!
                        for conList in contexts[context][word]:
                            for conList2 in contexts[context][word2]:
                                if word != word2: #removing diagonals
                                    # get intersections between these lists
                                    commonWords = [w for w in conList if w in set(conList2)]
                                    # increase the index of word and word2 by the number of intersections between their contexts

                                    # normalizing: co-occurrence equals the number of shared content words / total number of co-occurrences
                                    val = Decimal(len(commonWords))/Decimal((n-1 + n)/2)

                                    # Currently using a hack (multipling by 10) to get this to save correctly
                                    # because something about pandas dataframes doesn't like decimals...
                                    Q[currInd][word][word2] += round(val * 10)

                                    # not normalized...
                                    #Q[currInd][word][word2] += len(commonWords)
                    except:
                        pass
        currInd +=1
    return Q















    #
