import os
from unsupervised import bagOfWords
from processing import getDocuments
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np


'''
Counting on an individual document basis so that stat tests can be performed
'''

#### DEFINE GROUPS AND GROUP VARIABLES ####
groups = ["AD","TD"]
paths = {"AD":'texts/ADHD_various_half/AD_4',"TD":'texts/ADHD_various_half/TD_4'}
documents, texts = [{"AD":[],"TD":[]} for _ in range(2)]
counts, words, DMN_inverse, tasknet_activation = [{"AD":{},"TD":{}} for _ in range(4)]
networks = ["DMN","SN","LECN","RECN"]

#### PROCESS DOCUMENTS ####
for group in groups:
    texts[group] = sorted([os.path.join(paths[group], fn) for fn in os.listdir(paths[group])])
    if len(texts[group]) > 80:
       texts[group] = texts[group][1:len(texts[group])]

    documents[group] = getDocuments(texts[group], 'none', False, texts[group])

#### BAG OF WORDS ####
    for text in texts[group]:
        DMN_inverse[group][text]= {"SN":0,"LECN":0,"RECN":0}
        tasknet_activation[group][text] = {"SN_RECN":0,"SN_LECN":0,"LECN_RECN":0,"SN_LECN_RECN":0}
        doc_dict = {text:documents[group][text]} #making single dict to get BoW code to work on individual text
        counts[group][text], reducedDocuments, words[group][text] = bagOfWords([text], doc_dict, False, 0, False, False)
        counts[group][text] = counts[group][text][0] #flattening
        count_list = counts[group][text]
        word_list = words[group][text] # for code simplification

        #### SORT BY ACTIVITY ####

        for word in word_list:
            # inverse activity with DMN
            for taskNet in range(1,len(networks)):
                if word[0] == 'a' or word[0] == 'b':
                    if word[taskNet] == 'c' or word[taskNet] == 'd':
                        DMN_inverse[group][text][networks[taskNet]] += 1
                if word[0] == 'c' or word[0] == 'd':
                    if word[taskNet] == 'a' or word[taskNet] == 'b':
                        DMN_inverse[group][text][networks[taskNet]] += 1
            # co-activation between task networks
            if word[1] == 'c' or word[1] == 'd':
                if word[2] == 'c' or word[2] == 'd':
                    tasknet_activation[group][text]["SN_LECN"] += 1
                if word[3] == 'c' or word[3] == 'd':
                    tasknet_activation[group][text]["SN_RECN"] += 1
                if word [2] == 'c' or word[2] == 'd':
                    if word[3] == 'c' or word[3] == 'd':
                        tasknet_activation[group][text]["SN_LECN_RECN"] += 1
                        if word[0] == 'c' or word[0] == 'd':
                            tasknet_activation[group][text]["DMN_SN_LECN_RECN"] += 1s
            if word [2] == 'c' or word[2] == 'd':
                if word[3] == 'c' or word[3] == 'd':
                    tasknet_activation[group][text]["LECN_RECN"] += 1



#### STATS TIME ####
grp_inverse = {"AD":{"SN":[],"LECN":[],"RECN":[]},"TD":{"SN":[],"LECN":[],"RECN":[]}}
grp_taskactive = {"AD":{"SN_LECN":[],"SN_RECN":[],"LECN_RECN":[],"SN_LECN_RECN":[],"DMN_SN_LECN_RECN":[]},"TD":{"SN_LECN":[],"SN_RECN":[],"LECN_RECN":[],"SN_LECN_RECN":[],"DMN_SN_LECN_RECN":[]}}
for group in groups:
    for text in texts[group]:
        for net in networks[1:4]:
            grp_inverse[group][net].append(DMN_inverse[group][text][net])
        grp_taskactive[group]["SN_LECN"].append(tasknet_activation[group][text]["SN_LECN"])
        grp_taskactive[group]["SN_RECN"].append(tasknet_activation[group][text]["SN_RECN"])
        grp_taskactive[group]["LECN_RECN"].append(tasknet_activation[group][text]["LECN_RECN"])
        grp_taskactive[group]["SN_LECN_RECN"].append(tasknet_activation[group][text]["SN_LECN_RECN"])
        grp_taskactive[group]["DMNSN_LECN_RECN"].append(tasknet_activation[group][text]["DMN_SN_LECN_RECN"])


#print np.sum(grp_taskactive["AD"]["SN_LECN_RECN"])
#print np.sum(grp_taskactive["TD"]["SN_LECN_RECN"])
#print np.sum(grp_taskactive["AD"]["SN_LECN"])


# independent t-tests
inverse_probs = []
for net in networks[1:4]:
    t, p = stats.ttest_ind(grp_inverse["AD"][net],grp_inverse["TD"][net])
    inverse_probs.append(p)

taskactive_probs = []
co_nets = ["SN_LECN","SN_RECN","LECN_RECN","SN_LECN_RECN","DMN_SN_LECN_RECN"]
for nets in co_nets:
    t, p = stats.ttest_ind(grp_taskactive["AD"][nets],grp_taskactive["TD"][nets])
    taskactive_probs.append(p)

# bonferroni correction
reject, pvals_corr, x, y = multipletests(inverse_probs+taskactive_probs,alpha=0.05,method='bonferroni')
print "Bonferroni Corrected Outcome:"
print ["DMN_SN","DMN_LECN","DMN_RECN"] + co_nets
print reject
print pvals_corr, "\n"




















#
