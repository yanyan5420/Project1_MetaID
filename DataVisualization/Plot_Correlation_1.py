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



def filter_correlation(data, correlation, threshold):
    """
    This function is to filter the correlation matrix based on a given threshold, 
    and place them into a dictionary.
    
    Parameters
    ==========
    data: data frame
        the data frame containing sample, feature and intensity
    correlation: data frame
        the correlation matrix among all the features.
    threshold: float
        the correlation threshold
    ==========
    
    Output: dict
        a dictionary containing each feature and its correspondong correlated features.
    """
    
    corr_arr = np.array(correlation)
    filter_corr = dict()
    
    i = 0
    for arr in corr_arr:
        key = data.columns[i]
        corr_features = dict()
        
        idx = np.where(arr>=threshold)[0]
        for p in idx:
            corr_features[data.columns[p]] = {'corr': arr[p], 'diff': abs(data.iloc[2,p] - data.iloc[2,i])}
        
        filter_corr[key] = corr_features
        
        i += 1
    
    return filter_corr



def get_points_list(data, filter_corr, a_feature):
    """
    This function is to get the points list based on a specific feature, 
    which containing the m/z value, the intensity value, the correlation coefficient and the differences.
    
    Parameters
    ==========
    data: data frame
        the data frame contains sample, feature and intensity values.
    filter_corr: dict
        the dictionary contains the correlation features and their correpsonding correlation values.
    a_feature: str
        the specific feature name
    """
    
    points_list = []

    for feature, value in filter_corr[a_feature].items():

        x = data.loc['mzmed', feature]
        y = data.loc['fimed', feature]
        z = value['corr']
        d = value['diff']
        points_list.append((x,y,z,d))
    
    return points_list



# get the data and correlation.
input_file = "LCMS_data.xlsx"
data, correlation = get_correlation_matrix(input_file)

threshold = 0.8
a_feature = "SLPOS_454.2918_1.9604"
filt_corr = filter_correlation(data, correlation, threshold)
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

fig.suptitle("Feature correlating with 454.2918_1.9604", y=0.96, fontsize=14, fontweight='bold')
ax.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
ax.set_xlabel("m/z", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax.set_ylabel("intensity", fontdict={'fontsize': 12, 'fontweight':'bold'})

ax.xaxis.set_tick_params(labelsize=12) 
ax.xaxis.set_ticks(list(range(420, 510, 20)))
ax.yaxis.set_tick_params(labelsize=12)
ax.set_ylim(0, round(max(map(lambda x: x[1], points_list)))+100000)


colorValues = np.array(list(map(lambda x: x[2], points_list)))
normalize = mcolors.Normalize(vmin=colorValues.min(), vmax=colorValues.max())
colormap = cm.rainbow

for x,y,z,d in points_list:   
    label = str("%.3f" % x)+'\n'+'('+str("%.2f" % z)+')'
    ax.annotate(label, (x, y), horizontalalignment='left', rotation=0, color=colormap(normalize(z)), fontsize=8)

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(colorValues)
clb = fig.colorbar(scalarmappaple, shrink=0.5, fraction=0.035, pad=0.04)
clb.ax.set_title('corr', fontdict={'fontsize':12, 'fontweight':'bold'})
clb.ax.set_yticklabels(["%.2f" % i for i in clb.get_ticks()], fontdict={'fontsize':12})

fig.savefig("SLPOS_454.2918_1.9604_0.8.png", dpi=300)
