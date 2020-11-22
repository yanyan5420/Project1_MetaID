#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import re
import matplotlib.backends.backend_pdf



def get_correlation_matrix(input_file):
    """
    This function is to read the input file into a data frame and get the correlation matrix.
    """
    
    data = pd.read_excel(input_file, index_col=0)
    correlation = data.iloc[4:,:].corr()
    
    return data, correlation



def filter_correlation(data, correlation, threshold, tolerance):
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
    tolerance: float
        the tolerance of retention time
    ==========
    """
    corr_arr = np.array(correlation)
    filter_corr = dict()
    
    i = 0
    for arr in corr_arr:
        key = data.columns[i]
        corr_features = dict()
        rt = data.iloc[1,i]
        
        idx = np.where(arr>=threshold)[0]
        for p in idx:
            if rt-tolerance <= data.iloc[1,p] <= rt+tolerance:
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
        the specific feature name.
    ==========
    """
    points_list = []

    for feature, value in filter_corr[a_feature].items():

        x = data.loc['mzmed', feature]
        y = data.loc['fimed', feature]
        z = value['corr']
        d = value['diff']
        points_list.append((x,y,z,d))
    
    return points_list
 


def construct_df(features, filter_corr):
    """
    This function is to construct a dataframe to store all inforamtion about each features and their relative features,
    and then save it to a csv file.
    
    Parameters
    ==========
    features: data frame
        a data frame storing the needed features.
    filter_corr: dict
        the dictionary contains the correlation features and their correpsonding correlation values.
    ==========
    """
    df_list = []
    
    for i in range(len(features)):
        df = pd.DataFrame.from_dict(filter_corr[features.iloc[i,0]], orient='index')
        df.reset_index(inplace=True)
        df.columns = ['corr_features', 'correlation', 'difference']
        df.insert(0, 'feature_name', [features.iloc[i,0]]*len(filter_corr[features.iloc[i,0]]))
        
        df_list.append(df)
    
    all_df = pd.concat(df_list)
    all_df.set_index(['feature_name', 'corr_features'], inplace=True)
    all_df.to_csv("correlation_features.csv", index_label=['feature_name', 'corr_features'])
    
    return all_df

 

# get the data and correlation.
input_file = "LCMS_data.xlsx"
data, correlation = get_correlation_matrix(input_file) 
filter_corr = filter_correlation(data, correlation, 0.8, 0.08)
features = pd.read_csv("features_SLPOS.csv") 
df = construct_df(features, filter_corr)
 

def plot_features(feature_name, points_list):
    """
    This function is to plot all the correlated features based on a given feature
    
    Parameters
    ==========
    feature_name: str
        the feature name.
    points_list: list
        a list containing the m/z value, the intensity value, the correlation coefficient and the differences.
    ==========
    """
    
    # set background and spines
    fig, ax = plt.subplots(figsize=(10,6.18))
    ax.grid(True, which='major', linestyle=':')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # set titles and labels for x and y axis
    name = re.findall('SLPOS_(.+)', feature_name)[0]
    fig.suptitle("Feature correlating with "+name, y=0.96, fontsize=14, fontweight='bold')
    ax.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
    ax.set_xlabel("m/z", fontdict={'fontsize': 12, 'fontweight':'bold'})
    ax.set_ylabel("intensity", fontdict={'fontsize': 12, 'fontweight':'bold'})

    # plot the vertical lines
    for p in points_list:  
        ax.vlines(p[0], 0, p[1], lw=1.5) 

    # annotate each line and set the colorbar
    colorValues = np.array(list(map(lambda x: x[2], points_list)))
    normalize = mcolors.Normalize(vmin=colorValues.min(), vmax=colorValues.max())
    colormap = cm.rainbow

    for x,y,z,d in points_list:  
        label = "%.3f" % x
    #     label = str("%.3f" % x)+'\n'+'('+str("%.2f" % z)+')'
        ax.annotate(label, (x, y),
                    horizontalalignment='left', rotation=0, color=colormap(normalize(z)), fontsize=8)

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(colorValues)
    clb = fig.colorbar(scalarmappaple, shrink=0.5, fraction=0.035, pad=0.04)
    clb.ax.set_title('corr', fontdict={'fontsize':12, 'fontweight':'bold'})
    clb.ax.set_yticklabels(["%.2f" % i for i in clb.get_ticks()], fontdict={'fontsize':12})

    ax.xaxis.set_tick_params(labelsize=12) 
    ax.yaxis.set_tick_params(labelsize=12)
    
    ax.set_xlim(right=ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    fig.savefig(feature_name+"_0.8.png", dpi=300)


# plot features
features_points = dict()
for i in range(len(features)):  
    the_feature = features.iloc[i,0]
    features_points[the_feature] = get_points_list(data, filter_corr, the_feature)

for f in features.feature:
    points_list = features_points[f]
    plot_features(f, points_list)



# plot detailed correlation
points_list = features_points['SLPOS_673.5917_10.7084']

# set background and spines
fig, ax = plt.subplots(figsize=(10,6.18))
ax.grid(True, which='major', linestyle=':')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# set titles and labels for x and y axis
fig.suptitle("Feature correlating with 673.5917_10.7084", y=0.96, fontsize=14, fontweight='bold')
ax.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
ax.set_xlabel("m/z", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax.set_ylabel("intensity", fontdict={'fontsize': 12, 'fontweight':'bold'})

# plot the vertical lines
for p in points_list:  
    ax.vlines(p[0], 0, p[1], lw=1.5) 

# annotate each line and set the colorbar
colorValues = np.array(list(map(lambda x: x[2], points_list)))
normalize = mcolors.Normalize(vmin=colorValues.min(), vmax=colorValues.max())
colormap = cm.rainbow

for x,y,z,d in points_list: 
    label = "%.3f" % x
#     label = str("%.3f" % x)+'\n'+'('+str("%.2f" % z)+')'
    ax.annotate(label, (x, y), horizontalalignment='left', rotation=0, color=colormap(normalize(z)), fontsize=8)

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(colorValues)
clb = fig.colorbar(scalarmappaple, shrink=0.5, fraction=0.035, pad=0.04)
clb.ax.set_title('corr', fontdict={'fontsize':12, 'fontweight':'bold'})
clb.ax.set_yticklabels(["%.2f" % i for i in clb.get_ticks()], fontdict={'fontsize':12})

ax.xaxis.set_tick_params(labelsize=12) 
ax.yaxis.set_tick_params(labelsize=12)

ax.set_xlim(0, right=ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))


axin1 = ax.inset_axes([0.05, 0.4, 0.45, 0.45])
for p in points_list:
    if p[0] < 350:
        axin1.vlines(p[0], 0, p[1], lw=1.5) 

for x,y,z,d in points_list: 
    if x < 350:
        label = "%.3f" % x
        axin1.annotate(label, (x, y), horizontalalignment='center', rotation=0, color=colormap(normalize(z)), fontsize=7)

axin1.xaxis.set_tick_params(labelsize=8) 
axin1.yaxis.set_tick_params(labelsize=8)
axin1.set_ylim(top=axin1.get_ylim()[1]+10000)
axin1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

ax.indicate_inset_zoom(axin1)


axin2 = ax.inset_axes([0.65, 0.2, 0.336, 0.2])
for p in points_list:
    if p[0] > 600:
        axin2.vlines(p[0], 0, p[1], lw=1.5) 

for x,y,z,d in points_list: 
    if x>600:
        label = "%.3f" % x
        axin2.annotate(label, (x, y), horizontalalignment='center', rotation=0, color=colormap(normalize(z)), fontsize=7)

axin2.xaxis.set_tick_params(labelsize=8) 
axin2.yaxis.set_tick_params(labelsize=8)
axin2.set_xlim(axin2.get_xlim()[0]-1, axin2.get_xlim()[1]+1)
axin2.set_ylim(top=axin2.get_ylim()[1]+100000)
axin2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

ax.indicate_inset_zoom(axin2)

fig.savefig("SLPOS_673.5917_10.7084_0.8_detailed.png", dpi=300)



# plot detailed correlation
points_list = features_points['SLPOS_820.7439_10.2783']

# set background and spines
fig1, ax1 = plt.subplots(figsize=(12,6.18))
ax1.grid(True, which='major', linestyle=':')
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# set titles and labels for x and y axis
fig1.suptitle("Feature correlating with 820.7439_10.2783 (m/z<=700)", y=0.96, fontsize=14, fontweight='bold')
ax1.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
ax1.set_xlabel("m/z", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax1.set_ylabel("intensity", fontdict={'fontsize': 12, 'fontweight':'bold'})

# plot the vertical lines
for p in points_list: 
    if p[0] <= 750:
        ax1.vlines(p[0], 0, p[1], lw=1.5) 

# annotate each line and set the colorbar
colorValues = np.array(list(map(lambda x: x[2], points_list)))
normalize = mcolors.Normalize(vmin=colorValues.min(), vmax=colorValues.max())
colormap = cm.rainbow

for x,y,z,d in points_list: 
    if x <= 750:
        label = "%.3f" % x
    #     label = str("%.3f" % x)+'\n'+'('+str("%.2f" % z)+')'
        ax1.annotate(label, (x, y), horizontalalignment='left', rotation=0, color=colormap(normalize(z)), fontsize=8)

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(colorValues)
clb = fig1.colorbar(scalarmappaple, shrink=0.5, fraction=0.035, pad=0.04)
clb.ax.set_title('corr', fontdict={'fontsize':12, 'fontweight':'bold'})
clb.ax.set_yticklabels(["%.2f" % i for i in clb.get_ticks()], fontdict={'fontsize':12})


ax1.xaxis.set_tick_params(labelsize=12) 
ax1.yaxis.set_tick_params(labelsize=12)
# ax.xaxis.set_ticks(list(range(420, 510, 20)))

ax1.set_xlim(right=ax1.get_xlim()[1])
ax1.set_ylim(0, ax1.get_ylim()[1])

fig1.savefig("SLPOS_820.7439_10.2783_0.8_detailed_1.png", dpi=300) 

# set background and spines
fig2, ax2 = plt.subplots(figsize=(12,6.18))
ax2.grid(True, which='major', linestyle=':')
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# set titles and labels for x and y axis
fig2.suptitle("Feature correlating with 820.7439_10.2783 (m/z>700)", y=0.96, fontsize=14, fontweight='bold')
ax2.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
ax2.set_xlabel("m/z", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax2.set_ylabel("intensity", fontdict={'fontsize': 12, 'fontweight':'bold'})

# plot the vertical lines
for p in points_list: 
    if p[0] > 750:
        ax2.vlines(p[0], 0, p[1], lw=1.5) 

# annotate each line and set the colorbar
colorValues = np.array(list(map(lambda x: x[2], points_list)))
normalize = mcolors.Normalize(vmin=colorValues.min(), vmax=colorValues.max())
colormap = cm.rainbow

for x,y,z,d in points_list: 
    if x > 750:
        label = "%.3f" % x
    #     label = str("%.3f" % x)+'\n'+'('+str("%.2f" % z)+')'
        ax2.annotate(label, (x, y), horizontalalignment='left', rotation=0, color=colormap(normalize(z)), fontsize=8)

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(colorValues)
clb = fig2.colorbar(scalarmappaple, shrink=0.5, fraction=0.035, pad=0.04)
clb.ax.set_title('corr', fontdict={'fontsize':12, 'fontweight':'bold'})
clb.ax.set_yticklabels(["%.2f" % i for i in clb.get_ticks()], fontdict={'fontsize':12})


ax2.xaxis.set_tick_params(labelsize=12) 
ax2.yaxis.set_tick_params(labelsize=12)
# ax.xaxis.set_ticks(list(range(420, 510, 20)))

ax2.set_xlim(767,ax2.get_xlim()[1])
ax2.set_ylim(0, ax2.get_ylim()[1])

fig2.savefig("SLPOS_820.7439_10.2783_0.8_detailed_2.png", dpi=300)

pdf = matplotlib.backends.backend_pdf.PdfPages("SLPOS_820.7439_10.2783_0.8_detailed.pdf")
for fig in [fig1, fig2]:
    pdf.savefig(fig)
pdf.close()
