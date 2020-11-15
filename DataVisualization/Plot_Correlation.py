#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors


def get_correlation_matrix(input_file):
    """
    This function is to read the input file into a data frame and get the correlation matrix.
    """
    
    data = pd.read_excel(input_file, index_col=0)
    correlation = data.iloc[4:,:].corr()
    
    return data, correlation



def filter_correlation(correlation, threshold):
    """
    This function is to filter the correlation matrix based on a given threshold, 
    and place them into a dictionary.
    
    Parameters
    ==========
    correlation: data frame
        the correlation matrix among all the features.
    threshold: float
        the correlation threshold
    ==========
    
    Output: dict
        a dictionary containing each feature and its correspondong correlated features.
    """
    
    filt_corr = dict()
    
    for i in range(len(correlation)):
        key = correlation.index[i]
        value = dict()

        for j in range(len(correlation)):
            c = correlation.iloc[i, j]
            if c!=1 and c>= threshold:
                value[correlation.columns[j]] = c

        filt_corr[key] = value
    
    return filt_corr



def get_points_list(data, filt_corr, a_feature):
    """
    This function is to get the points list based on a specific feature, 
    which containing the m/z value, the intensity value, and the correlation value with the specific feature.
    
    Parameters
    ==========
    data: data frame
        the data frame contains sample, feature and intensity values.
    filt_corr: dict
        the dictionary contains the correlation features and their correpsonding correlation values.
    a_feature: str
        the specific feature name
    """
    
    # get the feature list which contains the specific feature and all the features associated with it.
    feature_list = [a_feature, ]
    for i in filt_corr[a_feature].keys():
        feature_list.append(i)

    feature_data = data.loc[:, feature_list]

    points_list = []
    for i in range(len(feature_data.columns)):
        # get the m/z value and maximum intensity value
        x = feature_data.iloc[2,i]
        y = max(feature_data[feature_data.columns[i]][4:])
        
        # get the correlation value
        if feature_data.columns[i] == a_feature:
            z = 1.0
        else:
            z = filt_corr[a_feature][feature_data.columns[i]]

        points_list.append((x,y,z))
    
    return points_list



# get the data and points list to plot later.
input_file = "LCMS_data.xlsx"
threshold = 0.8
a_feature = "SLPOS_454.2918_1.9604"

data, correlation = get_correlation_matrix(input_file)
filt_corr = filter_correlation(correlation, threshold)
points_list = get_points_list(data, filt_corr, a_feature)


# Start plotting
fig, ax = plt.subplots(figsize=(10,6.18))
ax.grid(True, which='major', linestyle=':')
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_position(('outward', 5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for p in points_list:  
    ax.vlines(p[0], 0, p[1], lw=1.5) 

ax.set_title("Feature correlating with 454.2918_1.9604 (C>0.8)", fontdict={'fontsize': 14, 'fontweight':'bold'})
ax.set_xlabel("m/z", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax.set_ylabel("intensity", fontdict={'fontsize': 12, 'fontweight':'bold'})

ax.xaxis.set_tick_params(labelsize=12) 
ax.xaxis.set_ticks(list(range(420, 510, 20)))
ax.yaxis.set_tick_params(labelsize=12)
ax.set_ylim(0, round(max(map(lambda x: x[1], points_list)))+1.5)


colorValues = np.array(list(map(lambda x: x[2], points_list)))
normalize = mcolors.Normalize(vmin=colorValues.min(), vmax=colorValues.max())
colormap = cm.rainbow

for x,y,z in points_list:   
    label = str("%.3f" % x)+'\n'+'('+str("%.2f" % z)+')'
    ax.annotate(label, (x, y), horizontalalignment='left', rotation=0, color=colormap(normalize(z)), fontsize=8)

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(colorValues)
clb = fig.colorbar(scalarmappaple, shrink=0.5, fraction=0.035, pad=0.04)
clb.ax.set_title('corr', fontdict={'fontsize':12, 'fontweight':'bold'})
clb.ax.set_yticklabels(["%.2f" % i for i in clb.get_ticks()], fontdict={'fontsize':12})

fig.savefig("SLPOS_454.2918_1.9604.png", dpi=300)

