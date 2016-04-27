import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import csv, json, os
from sklearn import manifold
from pandas.io.json import json_normalize
from scipy import stats
from unsupervised import bagOfWords
from unsupervised import runBag
from processing import getDocuments

n_feats = 20
top_features = [u'bbba_babb', u'aaaa_abaa', u'bbab_bbab', u'baaa_abbb', u'abbb_bbbb', u'aaaa_abbb', u'baab_aabb', u'babb_aaba', u'bbab_abab', u'abab_aaaa', u'abbb_abaa', u'abbb_babb', u'abab_abab', u'baab_aaba', u'baab_baaa', u'babb_bbaa', u'aaba_aaba', u'baab_abab', u'babb_aabb', u'abab_bbbb', u'bbbb_abab', u'abbb_aaaa', u'aaba_bbbb', u'aaaa_baaa', u'aaaa_baba', u'babb_bbab', u'abbb_aabb', u'aabb_abaa', u'aaab_baba', u'baba_abba', u'babb_aaaa', u'abbb_abba', u'babb_aaab', u'aaab_bbaa', u'bbab_bbbb', u'bbbb_baba', u'bbbb_abbb', u'baba_bbab', u'aaba_abbb', u'bbbb_bbbb', u'baaa_baaa', u'baab_aaaa', u'bbba_aaaa', u'aaab_abbb', u'abbb_baab', u'aaaa_abba', u'bbbb_bbab', u'abba_babb', u'babb_baab', u'aaab_abaa']
top_features = top_features[:n_feats]

path = 'texts/multiple_sites_full_2letter/NYU/both'
word_counts, feature_names = runBag(path)

ind_dict = dict((k,i) for i,k in enumerate(feature_names))
inter = set(ind_dict).intersection(top_features)
indices = [ind_dict[x] for x in inter]

feature_counts = []
for sub in word_counts:
    sub_counts = [sub[x] for x in indices]
    feature_counts.append(sub_counts)

feature_dict = {}
for word in top_features:
    feature_dict[word] = []
for sub in feature_counts:
    for count in range(len(sub)):
        feature_dict[top_features[count]].append(sub[count])

labels = ['AD'] * 40 + ['TD'] * 40
feature_dict['group'] = labels

feature_df = pd.DataFrame.from_dict(feature_dict)
feature_df = feature_df[top_features + ['group']]

td_sum = feature_df[feature_df['group'] == 'TD'].sum(numeric_only=True)
ad_sum = feature_df[feature_df['group'] == 'AD'].sum(numeric_only=True)
td_sum = pd.Series.to_frame(td_sum)
ad_sum = pd.Series.to_frame(ad_sum)
summed_feats = ad_sum.merge(td_sum,how='inner',right_index=True,left_index=True)
summed_feats.columns = ['AD', 'TD']
print summed_feats

ad_vals = [6.0, 44.0, 27.0, 36.0, 69.0, 54.0, 21.0, 20.0, 16.0, 18.0, 29.0, 20.0, 17.0, 45.0, 29.0, 15.0, 30.0, 16.0, 24.0, 9.0, 17.0, 39.0, 50.0, 40.0, 11.0, 28.0, 29.0, 14.0, 5.0, 15.0, 35.0, 43.0, 12.0, 19.0, 37.0, 27.0, 44.0, 13.0, 34.0, 83.0, 57.0, 39.0, 34.0, 16.0, 32.0, 30.0, 54.0, 20.0, 19.0, 14.0]
td_vals = [18.0, 17.0, 49.0, 66.0, 39.0, 28.0, 26.0, 33.0, 27.0, 27.0, 49.0, 37.0, 21.0, 23.0, 50.0, 26.0, 35.0, 31.0, 11.0, 20.0, 29.0, 35.0, 43.0, 55.0, 34.0, 21.0, 43.0, 25.0, 23.0, 24.0, 23.0, 46.0, 19.0, 6.0, 41.0, 34.0, 55.0, 32.0, 35.0, 52.0, 73.0, 27.0, 26.0, 25.0, 44.0, 16.0, 34.0, 29.0, 36.0, 14.0]


#sns.lmplot(data=feature_df,x='baaa_abbb',y='abbb_bbbb')
#plt.show()

# g = sns.PairGrid(feature_df, diag_sharey=False)
# g.map_lower(sns.kdeplot, cmap="Blues_d")
# g.map_upper(plt.scatter)
# g.map_diag(sns.kdeplot, lw=3)


summed_feats['words'] = summed_feats.index
summed_feats = pd.melt(summed_feats,id_vars='words')
sns.barplot(data=summed_feats,y='words',x='value',hue='variable')
plt.legend()
plt.show()

melted_df = pd.melt(feature_df, id_vars='group')
sns.violinplot(data=melted_df,y='variable',x='value',hue='group',cut=0,inner="points")
plt.legend()
plt.xlim(0,6)
plt.show()

sns.barplot(data=melted_df,y='variable',x='value',hue='group',linewidth=0.5)
plt.legend()
plt.xlim(0,3)
plt.show()
