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


def makeFeatureDF():
    top_features = [u'bbba_babb', u'aaaa_abaa', u'bbab_bbab', u'baaa_abbb', u'abbb_bbbb', u'aaaa_abbb', u'baab_aabb', u'babb_aaba', u'bbab_abab', u'abab_aaaa', u'abbb_abaa', u'abbb_babb', u'abab_abab', u'baab_aaba', u'baab_baaa', u'babb_bbaa', u'aaba_aaba', u'baab_abab', u'babb_aabb', u'abab_bbbb', u'bbbb_abab', u'abbb_aaaa', u'aaba_bbbb', u'aaaa_baaa', u'aaaa_baba', u'babb_bbab', u'abbb_aabb', u'aabb_abaa', u'aaab_baba', u'baba_abba', u'babb_aaaa', u'abbb_abba', u'babb_aaab', u'aaab_bbaa', u'bbab_bbbb', u'bbbb_baba', u'bbbb_abbb', u'baba_bbab', u'aaba_abbb', u'bbbb_bbbb', u'baaa_baaa', u'baab_aaaa', u'bbba_aaaa', u'aaab_abbb', u'abbb_baab', u'aaaa_abba', u'bbbb_bbab', u'abba_babb', u'babb_baab', u'aaab_abaa']

    path = 'texts/multiple_sites_full_2letter/NYU/both'
    word_counts, feature_names = runBag(path)

    ind_dict = dict((k,i) for i,k in enumerate(feature_names))
    inter = list(set(ind_dict).intersection(top_features))
    inter.sort(key=lambda x: top_features.index(x))
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

    return feature_df

def countFeatures(top_features):

    feature_df = makeFeatureDF()

    td_sum = feature_df[feature_df['group'] == 'TD'].sum(numeric_only=True)
    ad_sum = feature_df[feature_df['group'] == 'AD'].sum(numeric_only=True)
    td_sum = pd.Series.to_frame(td_sum)
    ad_sum = pd.Series.to_frame(ad_sum)
    summed_feats = ad_sum.merge(td_sum,how='inner',right_index=True,left_index=True)
    summed_feats.columns = ['AD', 'TD']

    summed_feats['words'] = summed_feats.index
    summed_feats = pd.melt(summed_feats,id_vars='words')

    melted_df = pd.melt(feature_df, id_vars='group')


    return summed_feats, melted_df

def drop_features(n_feats):
    top_features = [u'bbba_babb', u'aaaa_abaa', u'bbab_bbab', u'baaa_abbb', u'abbb_bbbb', u'aaaa_abbb', u'baab_aabb', u'babb_aaba', u'bbab_abab', u'abab_aaaa', u'abbb_abaa', u'abbb_babb', u'abab_abab', u'baab_aaba', u'baab_baaa', u'babb_bbaa', u'aaba_aaba', u'baab_abab', u'babb_aabb', u'abab_bbbb', u'bbbb_abab', u'abbb_aaaa', u'aaba_bbbb', u'aaaa_baaa', u'aaaa_baba', u'babb_bbab', u'abbb_aabb', u'aabb_abaa', u'aaab_baba', u'baba_abba', u'babb_aaaa', u'abbb_abba', u'babb_aaab', u'aaab_bbaa', u'bbab_bbbb', u'bbbb_baba', u'bbbb_abbb', u'baba_bbab', u'aaba_abbb', u'bbbb_bbbb', u'baaa_baaa', u'baab_aaaa', u'bbba_aaaa', u'aaab_abbb', u'abbb_baab', u'aaaa_abba', u'bbbb_bbab', u'abba_babb', u'babb_baab', u'aaab_abaa']
    top_features = top_features[:n_feats]
    return top_features

def wordFrequency():
    words = []
    top_features = [u'bbba_babb', u'aaaa_abaa', u'bbab_bbab', u'baaa_abbb', u'abbb_bbbb', u'aaaa_abbb', u'baab_aabb', u'babb_aaba', u'bbab_abab', u'abab_aaaa', u'abbb_abaa', u'abbb_babb', u'abab_abab', u'baab_aaba', u'baab_baaa', u'babb_bbaa', u'aaba_aaba', u'baab_abab', u'babb_aabb', u'abab_bbbb', u'bbbb_abab', u'abbb_aaaa', u'aaba_bbbb', u'aaaa_baaa', u'aaaa_baba', u'babb_bbab', u'abbb_aabb', u'aabb_abaa', u'aaab_baba', u'baba_abba', u'babb_aaaa', u'abbb_abba', u'babb_aaab', u'aaab_bbaa', u'bbab_bbbb', u'bbbb_baba', u'bbbb_abbb', u'baba_bbab', u'aaba_abbb', u'bbbb_bbbb', u'baaa_baaa', u'baab_aaaa', u'bbba_aaaa', u'aaab_abbb', u'abbb_baab', u'aaaa_abba', u'bbbb_bbab', u'abba_babb', u'babb_baab', u'aaab_abaa']
    path = 'texts/multiple_sites_full_2letter/NYU/both'
    word_counts, feature_names = runBag(path)

    for feat in feature_names:
        words.append(feat[0:4])

    words = sorted(list(set(words)))
    occ_mat = pd.DataFrame(columns=words, index=words)
    ad_mat = pd.DataFrame(columns=words, index=words)
    td_mat = pd.DataFrame(columns=words, index=words)

    feature_df = makeFeatureDF()
    for feat in top_features:
        occ_mat.loc[feat[0:4], feat[5:9]] = feature_df[feat].mean()
        ad_mat.loc[feat[0:4], feat[5:9]] = feature_df[feature_df['group'] == 'AD' ][feat].mean()
        td_mat.loc[feat[0:4], feat[5:9]] = feature_df[feature_df['group'] == 'TD' ][feat].mean()

    occ_mat = occ_mat.fillna(0)
    ad_mat = ad_mat.fillna(0)
    td_mat = td_mat.fillna(0)

    #normalizing (now matrix of joint probabilities)
    normed_ad_mat = ad_mat.divide((ad_mat.sum(axis=0)).sum())
    normed_td_mat = td_mat.divide((td_mat.sum(axis=0)).sum())

    def conditionalMatrix(ax, occurrence, normed):
        p = occurrence.sum(axis=ax).divide(occurrence.sum(axis=ax).sum())
        if ax == 0:
            cond_p = normed.divide(p,axis=1)
        else:
            cond_p = normed.divide(p,axis=0)
        cond_p = cond_p.fillna(0)
        return cond_p

    ad_second_given_first = conditionalMatrix(0,ad_mat,normed_ad_mat)
    ad_first_given_second = conditionalMatrix(1,ad_mat,normed_ad_mat)
    td_second_given_first = conditionalMatrix(0,td_mat,normed_td_mat)
    td_first_given_second = conditionalMatrix(1,td_mat,normed_td_mat)
    conditionals = [ad_second_given_first,ad_first_given_second,td_second_given_first,td_first_given_second]


    return normed_ad_mat, normed_td_mat, conditionals
