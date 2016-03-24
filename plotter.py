'''
Cannot run in Gensim virtual env
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nTopics = 20 #20

dynamic_topic_data = {}
for t in range(nTopics):
    #filename = ('dynamic_topics/20topics/dynamic_data_topic_%i')%(t)
    filename = ('dynamic_topics/20_topics/dynamic_data_topic_%i')%(t)
    dynamic_topic_data[t] = pd.DataFrame.from_csv(filename)

    p = sns.heatmap(dynamic_topic_data[t])

    title = "Topic %i" %(t)

    for label in p.get_xticklabels():
        label.set_rotation(40)
    for label in p.get_yticklabels():
        label.set_rotation(360)
    sns.plt.title(title)

    plt.show()
