import numpy as np
import pandas as pd
from decimal import *


'''
this will be where all occurrence counting functions go
so like bag of words, co-occurrence calculation, etc
'''

def QbyContext(documents, contexts, n):
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

        # normalize it. currently very small...
        for row in unique_words:
            for col in unique_words:
                Q[doc][row][col] = Decimal(Q[doc][row][col])/Decimal((len(unique_words) * (len(unique_words)+1))/ 2)
    return Q
