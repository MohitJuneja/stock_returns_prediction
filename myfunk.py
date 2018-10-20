import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
import pandas_profiling

from pylab import rcParams

#plt.rcParams['figure.figsize'] = 20,4
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 3
sns.set(style="darkgrid", color_codes=True)




def analyze_stock(asset):
    df = market_train_df.loc[market_train_df.assetCode==asset].set_index('time')
    fig = plt.figure(figsize=(20,20))
    plt.subplot(611)
    plt.plot(df.index.values, df.open, color='orange', linewidth=3, label='Open')
    plt.plot(df.index.values, df.close, color='royalblue', linewidth=2, linestyle='dashed', label='Close', alpha=0.5)
    plt.legend()
    plt.subplot(612)
    plt.plot(df.index.values, df.returnsClosePrevRaw1, color='orange', linewidth=2, label='returnsClosePrevRaw1')
    plt.plot(df.index.values, df.returnsClosePrevMktres1, color='royalblue', linewidth=1, linestyle='dashed', label='returnsClosePrevMktres1', alpha=0.5)
    plt.legend()
    plt.subplot(613)
    plt.plot(df.index.values, df.returnsOpenPrevRaw1, color='orange', linewidth=2, label='returnsOpenPrevRaw1')
    plt.plot(df.index.values, df.returnsOpenPrevMktres1, color='royalblue', linewidth=1, linestyle='dashed', label='returnsOpenPrevMktres1', alpha=0.5)
    plt.legend()
    plt.subplot(614)
    plt.plot(df.index.values, df.returnsClosePrevRaw10, color='orange', linewidth=2, label='returnsClosePrevRaw10')
    plt.plot(df.index.values, df.returnsClosePrevMktres10, color='royalblue', linewidth=1, linestyle='dashed', label='returnsClosePrevMktres10', alpha=0.5)
    plt.legend()
    plt.subplot(615)
    plt.plot(df.index.values, df.returnsOpenPrevRaw10, color='orange', linewidth=2, label='returnsOpenPrevRaw10')
    plt.plot(df.index.values, df.returnsOpenPrevMktres10, color='royalblue', linewidth=1, linestyle='dashed', label='returnsOpenPrevMktres10', alpha=0.5)
    plt.legend()
    plt.subplot(615)
    plt.plot(df.index.values, df.returnsOpenNextMktres10, color='orange', linewidth=2, label='returnsOpenNextMktres10')
    plt.legend()




# Takes a dataframe and the columnns as labels that will log transform them.
def log_transform(df,labels):
    for label in labels:
        new_label = label+'Log'
        a = np.min([0.001, int(np.floor(df[[label]].min().values[0]))])
        df[new_label] = df[[label]].apply(lambda x: np.log(x+1-a))
#        df.drop(label, inplace=True, axis=1)
    return df

# PLOTS Preprocessing data labels in bone structure
def print_preprocessing(df, labels, bone):
    fig = plt.figure(figsize=(20,20))
    s_param = [[len(bone), bone[row], row*bone[row]+col+1] for row in range(len(bone)) for col in range(bone[row])]
    for n_graph in range(np.sum(bone)):
        plt.subplot(s_param[n_graph][0], s_param[n_graph][1], s_param[n_graph][2])
        sns.boxplot(data=df.loc[:,labels[n_graph]], orient="h")
        plt.title(labels[n_graph])
    plt.show()
