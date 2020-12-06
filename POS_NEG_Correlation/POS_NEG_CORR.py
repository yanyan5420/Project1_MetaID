#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re


# import Positive data and Negative data
data1 = pd.read_excel("LC-MS_annotations_AWI.xlsx", sheet_name="SLPOS")
data2 = pd.read_excel("LC-MS_annotations_AWI.xlsx", sheet_name="SLNEG")


#========== 1. Use Positive Datasets to Annotate Negative Datasets

# filter features with n.d.
data1 = data1[data1["AIRWAVE1_MZRT_str"] != 'n.d.']
# filter metabolites
pos_data = data1[data1['RefMet_Standardized_name '].str.startswith(("CAR", "TG", "DG", 'MG'))==False]

# filter features with n.d.
data2 = data2[data2["AIRWAVE1_MZRT_str"] != 'n.d.']

join_data = pos_data.loc[:,['Metabolite','IonType','AIRWAVE1_MZRT_str']]

annotate_dict = dict(zip(pos_data.AIRWAVE1_MZRT_str, pos_data.Metabolite))

pos_features = pos_data.AIRWAVE1_MZRT_str
neg_features = data2.AIRWAVE1_MZRT_str

## Calculate Correlation
pos_samples = pd.read_excel("POS_samples.xlsx")
pos_samples.set_index(['Unnamed: 0'], inplace=True)
neg_samples = pd.read_excel("NEG_samples.xlsx")
neg_samples.set_index(['Unnamed: 0'], inplace=True)

pos_samples_df = pos_samples[[x for x in pos_features if x in pos_samples.columns]]
neg_samples_df = neg_samples[[x for x in neg_features if x in neg_samples.columns]]

pos_samples_arr = pos_samples_df.T.to_numpy()
neg_samples_arr = neg_samples_df.T.to_numpy()
both_samples_arr = np.concatenate((pos_samples_arr, neg_samples_arr))

corr_arr = np.corrcoef(both_samples_arr)[:len(pos_samples_arr),len(pos_samples_arr):]

corr_df = pd.DataFrame(corr_arr)
corr_df.index = pos_samples_df.columns
corr_df.columns = neg_samples_df.columns


corr_dict = {}
i = 0
for arr in corr_arr:
    
    inner_dict = {}
    
    pos_f = corr_df.index[i]
    pos_rt = float(re.findall(r'.+\_.+\_(.+)', pos_f)[0])
    
    idx = np.where(arr>=0.75)[0]
    
    for j in idx:
        neg_f = corr_df.columns[j]
        neg_rt = float(re.findall(r'.+\_.+\_(.+)', neg_f)[0])
        if pos_rt-0.04 <= neg_rt <= pos_rt+0.04:
            inner_dict[neg_f] = arr[j]
    
    if inner_dict != {}:
        corr_dict[pos_f] = inner_dict
    
    i += 1


neg_list = []
cor_list = []
for d in list(corr_dict.values()):
    for k, v in d.items():
        neg_list.append(k)
        cor_list.append(v)

pos_list = []
for key, value in corr_dict.items():
    
    pos_list = pos_list + [key]*len(value)

df = pd.DataFrame()
df['POS'] = pos_list
df['NEG'] = neg_list
df['CORR'] = cor_list

pos_df = join_data.merge(df, left_on='AIRWAVE1_MZRT_str', right_on='POS').drop('AIRWAVE1_MZRT_str',1)
pos_df.to_excel("POS_NEG_CORR.xlsx", index=False)


#========== 2. Use Negative Datasets to Annotate Positive Datasets

# filter features with n.d.
data1 = data1[data1["AIRWAVE1_MZRT_str"] != 'n.d.']
# filter features with n.d.
data2 = data2[data2["AIRWAVE1_MZRT_str"] != 'n.d.']
# filter metabolites
neg_data = data2[data2['RefMet_Main_class']!='Fatty acids']

join_data2 = neg_data.loc[:,['Metabolite','IonType','AIRWAVE1_MZRT_str']]

pos_features = data1.AIRWAVE1_MZRT_str
neg_features = neg_data.AIRWAVE1_MZRT_str

pos_samples_df = pos_samples[[x for x in pos_features if x in pos_samples.columns]]
neg_samples_df = neg_samples[[x for x in neg_features if x in neg_samples.columns]]

pos_samples_arr = pos_samples_df.T.to_numpy()
neg_samples_arr = neg_samples_df.T.to_numpy()
both_samples_arr = np.concatenate((pos_samples_arr, neg_samples_arr))

corr_arr = np.corrcoef(both_samples_arr)[len(pos_samples_arr):,:len(pos_samples_arr)]

corr_df = pd.DataFrame(corr_arr)
corr_df.index = neg_samples_df.columns
corr_df.columns = pos_samples_df.columns

corr_dict = {}
i = 0
for arr in corr_arr:
    
    inner_dict = {}
    
    neg_f = corr_df.index[i]
    neg_rt = float(re.findall(r'.+\_.+\_(.+)', neg_f)[0])
    
    idx = np.where(arr>=0.75)[0]
    
    for j in idx:
        pos_f = corr_df.columns[j]
        pos_rt = float(re.findall(r'.+\_.+\_(.+)', pos_f)[0])
        if neg_rt-0.04 <= pos_rt <= neg_rt+0.04:
            inner_dict[pos_f] = arr[j]
    
    if inner_dict != {}:
        corr_dict[neg_f] = inner_dict
    
    i += 1

neg_list = []
for key, value in corr_dict.items():
    
    neg_list = neg_list + [key]*len(value)

pos_list = []
cor_list = []
for d in list(corr_dict.values()):
    for k, v in d.items():
        pos_list.append(k)
        cor_list.append(v)

df = pd.DataFrame()
df['NEG'] = neg_list
df['POS'] = pos_list
df['CORR'] = cor_list

neg_df = join_data2.merge(df, left_on='AIRWAVE1_MZRT_str', right_on='NEG').drop('AIRWAVE1_MZRT_str',1)
neg_df.to_excel("NEG_POS_CORR.xlsx", index=False)
