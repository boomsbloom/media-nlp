'''
Cannot run in Gensim virtual env
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
#from unsupervised import jsd

nTopics = 1 #20 #5 #10
nDocuments = 40
nTimepoints = 164
#group = ['_TD','_AD'] #'' #'_AD'
#group = ['']
group = ['_network-wise']
#def plot_divergence():

def loadCSV(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        freqs = list(reader)

    freqs[0] = [float(val) for val in freqs[0]]

    return freqs

def plot_BoW():
    '''
    need to normalize
    '''
    AD_freqs = loadCSV("AD_meanBoW_nowindow")
    TD_freqs = loadCSV("TD_meanBoW_nowindow")

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

    top5_RF = ['bbac','bccc','aaba','adca','dbdb']
    #AD_freqs[0] = AD_freqs[0][0:100]
    #TD_freqs[0] = TD_freqs[0][0:30]

    def topFeatCount(freqs,label):
        f1 = []
        f2 = []
        for i in range(len(freqs[1])):
            if freqs[1][i] in top5_RF:
                f1.append(freqs[0][i])
                f2.append(freqs[1][i])
        f3 = [label] * len(f1)
        return [f1, f3, f2]

    topRF = [[]] * 2
    topRF[0] = topFeatCount(AD_freqs,'AD')
    topRF[1] = topFeatCount(TD_freqs,'TD')
    topRF = [topRF[0][0] + topRF[1][0], topRF[0][1] + topRF[1][1], topRF[0][2] + topRF[1][2]]
    topRF = np.asarray(topRF)
    #print topRF
    #topRF[1] = [string.astype(np.str_,16) for string in topRF[1]]
    #topRF[2] = [string.astype(np.str_,16) for string in topRF[2]]
    #print dt
    top_freqCounts = pd.DataFrame(data=np.transpose(topRF),index=topRF[2],columns=['count','group','word'])
    top_freqCounts["word"] = top_freqCounts["word"].astype('category')
    top_freqCounts["group"] = top_freqCounts["group"].astype('category')
    top_freqCounts["count"] = top_freqCounts["count"].astype('float')
    print top_freqCounts

    ax = sns.barplot(x="word", y="count", hue="group", data=top_freqCounts)
    plt.show()


    #fig, ax = plt.subplots()
    #axes = fig.gca()
    #index = np.arange(len(TD_freqs[0]))
    #ax.bar(index,TD_freqs[0],color='r') #s=20 should be abstracted to number of words in topics
    #bar_width = 0.35
    #plt.xticks(index + bar_width, (map(str,range(len(TD_freqs[0])))))
    #ax.set_xticklabels(TD_freqs[1])
    #for label in ax.get_xticklabels():
    #   label.set_rotation(90)
    #axes.set_ylim([0,100]) #0, 550

    #index = np.arange(len(AD_freqs[0]))
    #ax.bar(index,AD_freqs[0]) #s=20 should be abstracted to number of words in topics
    #plt.xticks(index + bar_width, (map(str,range(len(AD_freqs[0])))))
    #ax.set_xticklabels(AD_freqs[1])
    #for label in ax.get_xticklabels():
    #   label.set_rotation(90)
    #sns.plt.title("Top 100 states for AD subjects")


    #plt.show()



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
