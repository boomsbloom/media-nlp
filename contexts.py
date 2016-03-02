'''
functions to retrieve n words in context for each word in a document
'''

def getKWIC(texts, ngrams, n):
    kwicdict = {}
    for text in texts:
        if text != 'texts/mPFC_ofMRI/.DS_Store':
            if ngrams[text] != []:
                keyindex = len(ngrams[text][0]) // int(n/2 + 0.5)
                dic = {}
                for k in ngrams[text]:
                    if k[keyindex] not in dic:
                        dic[k[keyindex]] = [k]
                    else:
                        dic[k[keyindex]].append(k)
                kwicdict[text] = dic
    return kwicdict


def getnGrams(texts, n, documents):
    ngrams = {}
    for doc in documents:
        if doc != 'texts/mPFC_ofMRI/.DS_Store':
        #print doc
            ngrams[doc] = ([documents[doc][i:i+n] for i in range(len(documents[doc])-(n-1))])
        # this is imperfect because it doesn't count context for the last n-1 entries
    return getKWIC(texts, ngrams, n)
