'''
Cannot run in Gensim virtual env
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from unsupervised import jsd

nTopics = 1 #20 #5 #10
nDocuments = 40
nTimepoints = 164
#group = ['_TD','_AD'] #'' #'_AD'
#group = ['']
group = ['_network-wise']
#def plot_divergence():


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
plot_dynamics()
#plot_dynamics_diff()


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
