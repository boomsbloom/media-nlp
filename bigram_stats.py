import os, csv
from unsupervised import bagOfWords
from processing import getDocuments
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np

def loadCSV(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        freqs = list(reader)

    freqs[0] = [float(val) for val in freqs[0]]

    return freqs

#### DEFINE GROUPS AND GROUP VARIABLES ####
groups = ["AD","TD"]
paths = {"AD":'texts/multiple_sites_full_2letter/NYU/AD_2',"TD":'texts/multiple_sites_full_2letter/NYU/TD_2'}
documents, texts, dmn_sn, dmn_recn = [{"AD":[],"TD":[]} for _ in range(4)]
counts, words = [{"AD":{},"TD":{}} for _ in range(2)]

#### PROCESS DOCUMENTS ####
#bigrams
vocab = loadCSV('NYU_AD_2letter_bigrams_BoW')
vocab = vocab[1]

#trigrams
# ad_voc = loadCSV('NYU_AD_2letter_trigrams_BoW')
# td_voc = loadCSV('NYU_TD_2letter_trigrams_BoW')
# ad_voc = ad_voc[1]
# td_voc = td_voc[1]
# vocab = [w for w in td_voc if w in set(ad_voc)]
# print vocab

vocab_dict = {"AD":dict((word,[]) for word in vocab),"TD":dict((word,[]) for word in vocab)}

for group in groups:
    texts[group] = sorted([os.path.join(paths[group], fn) for fn in os.listdir(paths[group])])
    if len(texts[group]) > 80:
       texts[group] = texts[group][1:len(texts[group])]

    documents[group] = getDocuments(texts[group], 'none', False, texts[group])

#### BAG OF WORDS ####
    for text in texts[group]:
        if '.DS_Store' not in text:
            doc_dict = {text:documents[group][text]} #making single dict to get BoW code to work on individual text
            counts[group][text], reducedDocuments, words[group][text] = bagOfWords([text], doc_dict, True, 0, False, False)
            counts[group][text] = counts[group][text][0] #flattening
            count_list = counts[group][text]
            word_list = words[group][text] # for code simplification

            no_occurrence = [w for w in vocab if w not in set(word_list)]
            occurrence = [w for w in vocab if w in set(word_list)]

            for word in occurrence:
                count_ind = word_list.index(word)
                vocab_dict[group][word].append(count_list[count_ind])
            for word in no_occurrence:
                vocab_dict[group][word].append(0)

            dmn_sn_counter = 0
            dmn_recn_counter = 0
            for word in word_list:
                if (word[0] == 'a' and word[1] == 'a') and (word[5] == 'a' and word[6] == 'b'):
                #if (word[0] == 'a' and word[1] == 'a' and word[2] == 'a' and word[3] == 'a'): #or (word[5] == 'a' and word[6] == 'a' and word[7] == 'a' and word[8] == 'a'):
                    dmn_sn_counter += 1
                if (word[0] == 'a' and word[1] == 'a') and (word[5] == 'a' and word[8] == 'b'):
                    dmn_recn_counter += 1

            dmn_sn[group].append(dmn_sn_counter)
            dmn_recn[group].append(dmn_recn_counter)

# # independent t-tests
bigram_probs = []
for word in vocab:
    t, p = stats.ttest_ind(vocab_dict["AD"][word],vocab_dict["TD"][word])
    bigram_probs.append(p)

dmn_sn_probs = []
dmn_recn_probs = []
t, dmn_sn_p = stats.ttest_ind(dmn_sn["AD"],dmn_sn["TD"])
#print "DMN/SN aa --> DMN/SN ab", "AD total:", np.sum(dmn_sn["AD"]), "TD total:", np.sum(dmn_sn["TD"]), "p-value:", dmn_sn_p

t, dmn_recn_p = stats.ttest_ind(dmn_recn["AD"],dmn_recn["TD"])
#print "DMN/RECN aa --> DMN/RECN ab", "AD total:", np.sum(dmn_recn["AD"]), "TD total:", np.sum(dmn_recn["TD"]), "p-value:", dmn_recn_p

#
#print bigram_probs
sigs = [p <= 0.05 for p in bigram_probs]
uncorrected_significant = []
for p in range(len(sigs)):
    if (sigs[p]):
        uncorrected_significant.append(vocab[p])
print uncorrected_significant
#

#reject, pvals_corr, x, y = multipletests(bigram_probs,alpha=0.05,method='bonferroni')
# print "Bonferroni Corrected Outcome:"
# print vocab
# print reject
# print pvals_corr, "\n"















#
