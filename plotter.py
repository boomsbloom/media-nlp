'''
Cannot run in Gensim virtual env
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import csv, json
from sklearn import manifold
from pandas.io.json import json_normalize
from scipy import stats

#from unsupervised import jsd

nTopics = 1 #20 #5 #10
nDocuments = 40
nTimepoints = 164
nModels = 5
#group = ['_TD','_AD'] #'' #'_AD'
#group = ['']
group = ['_network-wise']
#def plot_divergence():

def plotTopFeatures():
    data = json.loads(open('featureSelection_nowindow.json').read())
    curDat = data["texts/ADHD_various_half/"]
    print curDat.keys()

    rf_features = {}
    for let in range(1,4): #len(curDat)
        unique = list(set(sum(curDat[str(let)]['features'], [])))
        curList = curDat[str(let)]['features']
        features = dict([(key, {'importance':0,'std':0}) for key in unique])

        for li in range(len(curList)):
            for word in curList[li]:
                index = curList[li].index(word)
                importance = curDat[str(let)]['importance'][li][index]
                #std = curDat[str(let)]['stds'][li][index]

                if word in unique:
                    features[word]['importance'] = features[word]['importance'] + importance
                    #features[word]['std'] = features[word]['std'] + std

        for i in range(len(features.values())):
            features.values()[i]['importance'] = features.values()[i]['importance']/nModels
            #features.values()[i]['std'] = features.values()[i]['std']/nModels

        rf_features[let+2] = features

        df = pd.DataFrame.from_dict(rf_features[let+2])
        df = df.transpose()

        fig = df.plot(x=df.index, y='importance', kind='bar')
        plt.show()


#plotTopFeatures()

def plotrfACC():
    #data = json.loads(open('rf_accs.json').read())
    data = json.loads(open('rf_accs_top3.json').read())
    data = json.loads(open('rf_accs_nowindow.json').read())
    nLetter = 3 #14
    data["texts/ADHD_various_half/"] = [data["texts/ADHD_various_half/"][i] for i in [1,2,3]]

    sns.set_style("dark")

    #f, (ax1, ax2) = plt.subplots(1, 2)
    f, ax1 = plt.subplots()
    bar1 = ax1.bar(range(nLetter),data["texts/ADHD_various_half/"])
    ax1.set_title('RF accs for half SAX')
    plt.sca(ax1)
    plt.xticks(np.arange(nLetter) + .4, range(3,nLetter+3))
    plt.xlabel('# of bins (letters)/word')
    ax1.set_ylim([0.6,0.9])

    #bar2 = ax2.bar(range(nLetter),data["texts/ADHD_various_full/"])
    #ax2.set_title('RF accs for full SAX')
    #plt.sca(ax2)
    #plt.xticks(np.arange(nLetter) + .4, range(2,nLetter+2))
    #plt.xlabel('# of bins (letters)/word')
    #ax2.set_ylim([0.6,0.9])

    plt.show()

#plotrfACC()

def mdsModel(sims):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(sims).embedding_
    #rescale data
    npos = mds.fit_transform(sims, init=pos)
    return npos

def word_dist(Q,C):
    letters = ['a','b','c','d']
    inner = 0
    for ind in range(len(Q)):
        for i in range(len(letters)):
            if Q[ind] == letters[i]:
                q = i
            if C[ind] == letters[i]:
                c = i
        inner+=(np.abs(q - c) * (0.67))**2
    return np.sqrt(inner)


def plotWordDist(words, diffs):
    dist_mat = np.zeros((len(words), len(words)))
    for word in words:
        for word2 in words:
            dist_mat[words.index(word),words.index(word2)] = word_dist(word,word2)

    npos = mdsModel(dist_mat)

    colors = ['r','b']
    groups = ['TD','AD']

    fig, ax = plt.subplots()
    for d in range(len(diffs)):
        if diffs[d] > 0:
            color = colors[1]
        else:
            color = colors[0]
        ax.scatter(npos[d, 0], npos[d, 1], s=len(words), c=color)

    for i, txt in enumerate(words):
        ax.annotate(txt, (npos[:, 0][i],npos[:, 1][i]))

    recs = []
    for i in range(0,len(groups)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
        ax.legend(recs,groups)

    sns.plt.title("Word similarity for AD top 30 states (color by larger count)")

    plt.show()


def plotActivationLevels():
    #AD_inverse = [653,807,478]
    #TD_inverse = [1085,907,1060]
    #activations = AD_inverse + TD_inverse
    #groups = ['AD'] * 3 + ['TD'] * 3
    #networks = ['DMN\nSN','DMN\nLCEN','DMN\nRCEN'] * 2
    AD_active = [441, 518, 588, 378, 277]
    TD_active = [472, 486, 329, 208, 27]
    activations = AD_active + TD_active
    groups = ['AD'] * 5 + ['TD'] * 5
    networks = ['SN\nLCEN','SN\nRCEN','LCEN\nRCEN','SN\nLCEN\nRCEN','All (task + DMN)'] * 2
    dataMat = [activations, groups, networks]

    activationMat = pd.DataFrame(data=np.transpose(dataMat),columns=['activations','groups','networks'])

    activationMat["activations"] = activationMat["activations"].astype('float')
    activationMat["groups"] = activationMat["groups"].astype('category')
    activationMat["networks"] = activationMat["networks"].astype('category')

    ax = sns.barplot(x="networks", y="activations", hue="groups", data=activationMat)
    #sns.plt.title("Count of inverse network activation with DMN\n (a/b = inactive | c/d = active)")
    sns.plt.title("Task network co-activation\n (c/d = active)")
    ax.set_ylim([0,600])
    #ax.set_ylim([400,1100])
    plt.show()

#plotActivationLevels()

def loadCSV(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        freqs = list(reader)

    freqs[0] = [float(val) for val in freqs[0]]

    return freqs


def plot_BoW():

    #AD_freqs = loadCSV("AD_meanBoW_nowindow")
    #TD_freqs = loadCSV("TD_meanBoW_nowindow")
    #AD_freqs = loadCSV("AD_BoW_only_bigrams")
    #TD_freqs = loadCSV("TD_BoW_only_bigrams")
    #AD_freqs = loadCSV("AD_BoW_FULL_only_bigrams")
    #TD_freqs = loadCSV("TD_BoW_FULL_only_bigrams")
    #AD_freqs = loadCSV('AD_4words_half')
    #TD_freqs = loadCSV('TD_4words_half')

    #AD_freqs = loadCSV('NYU_AD_2letter_bigrams_BoW')
    #TD_freqs = loadCSV('NYU_TD_2letter_bigrams_BoW')
    AD_freqs = loadCSV('NYU_AD_2letter_trigrams_BoW')
    TD_freqs = loadCSV('NYU_TD_2letter_trigrams_BoW')


    #AD_freqs = loadCSV('AD_5letter_half')
    #TD_freqs = loadCSV('TD_5letter_half')

    floor = 0

    def topFreqs(freqs):
        l1 = []
        l2 = []
        for c in range(len(freqs[0])):
            if freqs[0][c] > floor: #5
                l1.append(freqs[0][c])
                l2.append(freqs[1][c])
        tFreqs = [l1, l2]

        indices = [i[0] for i in sorted(enumerate(tFreqs[0]), key=lambda x:x[1], reverse=True)]
        sort = [i[1] for i in sorted(enumerate(tFreqs[0]), key=lambda x:x[1], reverse=True)]
        sortNames = [tFreqs[1][ind] for ind in indices]

        return [sort, sortNames]

    AD_freqs = topFreqs(AD_freqs)
    TD_freqs = topFreqs(TD_freqs)

    #print AD_freqs
    #print TD_freqs
    #AD_freqs[0] = [count/(np.sum(AD_freqs[0])) for count in AD_freqs[0]] #normalizing
    #TD_freqs[0] = [count/(np.sum(TD_freqs[0])) for count in TD_freqs[0]] #normalizing

    #top_RF = ['aaaa','cccc','bbca','abac','cabb']
    #top_RF = ['bbac','bccc','aaba','dbdb','adca']
    #top_RF = ['eeee','aaba','abaa','bccc','deee']


    word_list = []
    #sub_list = ['bbcb','bccc','bccb','bbcc','bbcb','','','','','']
    #uncorrected_significant = ['aaaa_abaa', 'aaaa_abab', 'aaaa_abba', 'aaaa_abbb', 'aaaa_baba', 'aaab_baba', 'aaab_babb', 'aaab_bbaa', 'aaba_babb', 'aabb_aaba', 'abab_bbbb', 'abba_aaaa', 'abbb_abaa', 'abbb_babb', 'abbb_bbbb', 'baaa_abbb', 'baab_aaba', 'baab_abab', 'baab_baaa', 'baba_aabb', 'baba_bbab', 'babb_aabb', 'babb_baab', 'bbab_bbab', 'bbba_babb', 'bbbb_abab', 'bbbb_bbab', 'bbbb_bbbb']
    uncorrected_significant = ['aaaa_aaaa_bbbb', 'aaaa_aaba_bbaa', 'aaaa_abab_abbb', 'aaaa_baaa_aaaa', 'aaaa_baba_abbb', 'aaba_baab_baab', 'aaba_baab_bbab', 'abaa_baaa_baab', 'abba_aaaa_abbb', 'abba_baaa_babb', 'abba_baab_baaa', 'abba_baab_bbbb', 'abbb_aaba_aaaa', 'abbb_abbb_aaaa', 'abbb_baaa_baba', 'abbb_bbbb_bbab', 'abbb_bbbb_bbba', 'baaa_abaa_bbab', 'baaa_abba_aaaa', 'baaa_abbb_aabb', 'baaa_baab_bbba', 'baaa_bbbb_abbb', 'baab_aaba_aaba', 'baab_aabb_bbbb', 'baba_aaaa_bbaa', 'baba_baaa_baaa', 'babb_bbaa_aaba', 'bbaa_abbb_bbbb', 'bbab_baaa_baaa', 'bbbb_aaaa_babb', 'bbbb_abbb_aabb', 'bbbb_abbb_bbbb', 'bbbb_baaa_bbbb', 'bbbb_bbab_aaaa', 'bbbb_bbbb_aaaa', 'bbbb_bbbb_abbb', 'bbbb_bbbb_baaa']
    for w in range(len(AD_freqs[0])):
        #if (TD_freqs[1][w][2] == 'c' or TD_freqs[1][w][2] == 'd') and (TD_freqs[1][w][3] == 'c' or TD_freqs[1][w][3] == 'd') and (TD_freqs[1][w][1] == 'c' or TD_freqs[1][w][1] == 'd')  and (TD_freqs[1][w][0] == 'c' or TD_freqs[1][w][0] == 'd'):
        #if (AD_freqs[1][w][0] == 'a' or AD_freqs[1][w][0] == 'b') and (AD_freqs[1][w][1] == 'c' or AD_freqs[1][w][1] == 'd'):
        #if (TD_freqs[1][w][0:4] == 'aaaa'):
        if TD_freqs[1][w] in uncorrected_significant:
            #if TD_freqs[1][w] in sub_list: #not in
            word_list.append(TD_freqs[1][w])
    print word_list

    def sortByWords(l):
        f1 = []
        f2 = []
        for w in range(len(l[1])):
            print l[1][w]
            if l[1][w] in word_list:
                f1.append(l[0][w])
                f2.append(l[1][w])
        return [f1, f2]

    #AD_freqs[0] = AD_freqs[0][0:30]
    #TD_freqs[0] = TD_freqs[0][0:30]#[0:30]#[0:30]
    AD_freqs = sortByWords(AD_freqs)
    TD_freqs = sortByWords(TD_freqs)

    def TDbyTopAD():
        f1 = []
        f2 = []
        for w in range(len(TD_freqs[1])):
            if TD_freqs[1][w] in AD_freqs[1]:
                f1.append(TD_freqs[0][w])
                f2.append(TD_freqs[1][w])
        return [f1, f2]

    def getSortedIndex(l1, l2):
        l2_count = []
        for w in range(len(l2)):
            for w2 in range(len(l1[1])):
                if l2[w] == l1[1][w2]:
                    l2_count.append(l1[0][w2])
        return [l2_count, l2]

    l1 = TDbyTopAD()
    l2 = sorted(l1[1],key=AD_freqs[1].index)
    TD_freqs = getSortedIndex(l1, l2)

    AD_freqs[0] = AD_freqs[0]#[0:30]
    TD_freqs[0] = TD_freqs[0]#[0:30]

    def topFeatCount(freqs,label):
        f1 = []
        f2 = []
        for i in range(len(freqs[1])):
            if freqs[1][i] in top_RF:
                f1.append(freqs[0][i])
                f2.append(freqs[1][i])
        f3 = [label] * len(f1)
        return [f1, f3, f2]

    # topRF = [[]] * 2
    # topRF[0] = topFeatCount(AD_freqs,'AD')
    # topRF[1] = topFeatCount(TD_freqs,'TD')
    # topRF = [topRF[0][0] + topRF[1][0], topRF[0][1] + topRF[1][1], topRF[0][2] + topRF[1][2]]
    # topRF = np.asarray(topRF)
    # top_freqCounts = pd.DataFrame(data=np.transpose(topRF),index=topRF[2],columns=['count','group','word'])
    # top_freqCounts["word"] = top_freqCounts["word"].astype('category')
    # top_freqCounts["group"] = top_freqCounts["group"].astype('category')
    # top_freqCounts["count"] = top_freqCounts["count"].astype('float')
    #
    # ax = sns.barplot(x="word", y="count", hue="group", data=top_freqCounts)
    # sns.plt.title("Word counts for top 5 RF features")
    # plt.show()

    #
    # fig, ax = plt.subplots()
    # axes = fig.gca()
    # index = np.arange(len(TD_freqs[0]))
    # td_bar = ax.bar(index,TD_freqs[0],color='r',alpha=0.7)
    # bar_width = 0.35
    # plt.xticks(index + bar_width, (map(str,range(len(TD_freqs[0])))))
    # ax.set_xticklabels(TD_freqs[1])
    # for label in ax.get_xticklabels():
    #   label.set_rotation(90)
    # axes.set_ylim([0,100]) #0, 550
    #
    # index = np.arange(len(AD_freqs[0]))
    # ad_bar = ax.bar(index,AD_freqs[0],alpha=0.7)
    # plt.xticks(index + bar_width, (map(str,range(len(AD_freqs[0])))))
    # ax.set_xticklabels(AD_freqs[1])
    # for label in ax.get_xticklabels():
    #   label.set_rotation(90)
    # sns.plt.title("AD states")
    #
    # ax.legend((td_bar[0], ad_bar[0]), ('TD', 'AD'))
    AD_freqs[0] = AD_freqs[0][0:200]
    TD_freqs[0] = TD_freqs[0][0:200]
    diffs = np.asarray(AD_freqs[0]) - np.asarray(TD_freqs[0])
    AD_count = 0
    TD_count = 0
    for val in diffs:
        if val < 0:
            TD_count += val
        else:
            AD_count += val
    print 'TD:', abs(TD_count), 'AD:', AD_count, 'Diff:', abs(abs(TD_count)-AD_count)
    # plotWordDist(TD_freqs[1], diffs)


    #
    fig, ax = plt.subplots()
    axes = fig.gca()
    index = np.arange(len(TD_freqs[0]))
    td_bar = ax.bar(index,diffs,color='g') #s=20 should be abstracted to number of words in topics
    bar_width = 0.35
    plt.xticks(index + bar_width, (map(str,range(len(TD_freqs[0])))))
    ax.set_xticklabels(TD_freqs[1])
    for label in ax.get_xticklabels():
      label.set_rotation(90)
    axes.set_ylim([-15,15]) #0, 550
    #sns.plt.title("AD Top 100 states by word count (AD - TD)")
    sns.plt.title("AD - TD")

    plt.show()

def plot_Networkletters(documents, title, letters):

    DMN = []
    SN = []
    LECN = []
    RECN = []
    for doc in documents:
        for word in documents[doc]:
            DMN.append(word[0])
            SN.append(word[1])
            LECN.append(word[2])
            RECN.append(word[3])

    letter_counts = {}
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    for let in letters:
        c1.append(DMN.count(let))
        c2.append(SN.count(let))
        c3.append(LECN.count(let))
        c4.append(RECN.count(let))

    letter_counts['DMN'] = c1
    letter_counts['SN'] = c2
    letter_counts['LECN'] = c3
    letter_counts['RECN'] = c4

    a = pd.DataFrame.from_dict(letter_counts)
    a.index = letters

    fig = a.plot(x=a.index, y=a.columns, kind='bar')
    fig.set_ylim([1000,1200])
    for label in fig.get_xticklabels():
      label.set_rotation(360)


    sns.plt.title(title)
    plt.show()

def plot_relevance():
    '''
    Relevance is the sum of topc proportions for each doc at t / num documents
    '''
    filename = "dynamic_topics/%i_topics_network-wise/gammas.csv"%(nTopics)
    gammas = np.loadtxt(open(filename,"rb"),delimiter=",")
    gammas = np.reshape(gammas,(nTimepoints,nDocuments,nTopics))

    summed_tprop = np.zeros((nTimepoints,nTopics))
    for j in range(nTopics):
        for t in range(nTimepoints):
            for s in range(nDocuments):
                summed_tprop[t,j] += gammas[t,s,j]
            summed_tprop[t,j] = summed_tprop[t,j]/nDocuments

        print np.mean([summed_tprop[t,j] for t in range(nTimepoints)], axis = 0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.stackplot(np.arange(nTimepoints), np.transpose(summed_tprop))
    ax.set_title('Distribution of Topic Relevance over Time')
    ax.set_ylabel('Relevance')
    ax.margins(0, 0) # Set margins to avoid "whitespace"

    plt.show()


def plot_dynamics():
    dynamic_topic_data = {}
    for t in range(nTopics):
        filename = ('dynamic_topics/%i_topics_TD%s/dynamic_data_topic_%i')%(nTopics,group[0],t)
        #filename = ('dynamic_topics/%i_topics%s/dynamic_data_topic_%i_1word')%(nTopics,group[0],t)
        dynamic_topic_data[t] = pd.DataFrame.from_csv(filename)

        p = sns.heatmap(dynamic_topic_data[t])

        title = "%s Topic %i" %(group[0],t)
        #title = "TD Single Topic Dynamics"

        for label in p.get_xticklabels():
            label.set_rotation(40)
        for label in p.get_yticklabels():
            label.set_rotation(360)
        sns.plt.title(title)

        plt.show()

def plot_dynamics_diff(): #currently only have data for 1 topic for TD - ADHD
    TD_dynamic_topic_data = {}
    AD_dynamic_topic_data = {}
    diff_dynamic_topic_data = {}
    for t in range(nTopics):
        TD_filename = ('dynamic_topics/%i_topics_TD_network-wise/dynamic_data_topic_%i')%(nTopics,t)
        #TD_filename = ('dynamic_topics/%i_topics%s/dynamic_data_topic_%i')%(nTopics,group[0],t)
        TD_dynamic_topic_data[t] = pd.DataFrame.from_csv(TD_filename)

        AD_filename = ('dynamic_topics/%i_topics_AD_network-wise/dynamic_data_topic_%i')%(nTopics,t)
        #AD_filename = ('dynamic_topics/%i_topics%s/dynamic_data_topic_%i')%(nTopics,group[1],t)
        AD_dynamic_topic_data[t] = pd.DataFrame.from_csv(AD_filename)

        TD_rows = list(TD_dynamic_topic_data[t].index)
        AD_rows = (AD_dynamic_topic_data[t].index)

        TD_only = list(set(TD_rows)-set(AD_rows))
        AD_only = list(set(AD_rows)-set(TD_rows))
        common_words = list(set(TD_rows).intersection(AD_rows))
        all_words = TD_only + AD_only + common_words

        diff_dynamic_topic_data[t] = pd.DataFrame(np.zeros(shape=(len(all_words),nTimepoints)),all_words,range(nTimepoints))
        for word in all_words:
            if word in common_words:
                x = TD_dynamic_topic_data[t].loc[word] - AD_dynamic_topic_data[t].loc[word]
            elif word in TD_only:
                x = TD_dynamic_topic_data[t].loc[word]
            else:
                x = -AD_dynamic_topic_data[t].loc[word]

            for val in range(len(x)):
                diff_dynamic_topic_data[t].loc[word][val] = x[val]


        p = sns.heatmap(diff_dynamic_topic_data[t])

        title = "TD - ADHD Topic %i Dynamics" %(t)

        for label in p.get_xticklabels():
            label.set_rotation(40)
        for label in p.get_yticklabels():
            label.set_rotation(360)
        sns.plt.title(title)

        plt.show()
        #print diff_dynamic_topic_data[t]
#plot_relevance()
#plot_dynamics()
#plot_dynamics_diff()
plot_BoW()


#plt.plot(summed_tprop)
#plt.show()
#gamma_mu = np.zeros((80, 20))
#for s in range(80):
#    for j in range(20):
#        gamma_mu[s,j] = np.mean([gammas[t,s,j] for t in range(164)], axis = 0)
#print max(gamma_mu)
#sns.heatmap(gamma_mu)
#plt.show()

#print gammas[163]
#sns.heatmap(gammas[160])
#plt.show()

#np.mean([list_of_arrays[t-j] for j in range1(1,N)], axis = 0)
