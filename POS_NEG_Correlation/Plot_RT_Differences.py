#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

data = pd.read_excel("pos_neg_overlap.xlsx")

# get RT of the positive data
data['RT_POS'] = data.AIRWAVE1_MZRT_str_x.apply(lambda x: float(re.findall(r'.+\_.+\_(.+)', x)[0]))
# get RT of the negative data
data['RT_NEG'] = data.AIRWAVE1_MZRT_str_y.apply(lambda x: float(re.findall(r'.+\_.+\_(.+)', x)[0]))
# calculate RT differences
data['RT_difference'] = data.RT_POS - data.RT_NE
# calculate RT abosulte differences
data['RT_DIFF'] = abs(data.RT_difference)

fig, ax = plt.subplots(figsize=(10,6.18))
ax.grid(True, which='major', linestyle=':')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

x = data.index
y = data.RT_difference
ax.plot(x, y, 'o', color='skyblue')
ax.plot(x, y, '-ok', color='darkblue')

for i, j in zip(x,y):
    
    if i % 2 == 0:
        label = data.iloc[i,0].split(' ')[-1]
        ax.annotate(label, (i,j), horizontalalignment='center', va='top', rotation=285, fontsize=10)
    else:
        label = data.iloc[i,0].split(' ')[-1]
        ax.annotate(label, (i,j), horizontalalignment='center', va='bottom', rotation=285, fontsize=10)
    
# set titles and labels for x and y axis
fig.suptitle("Retention Time Differences between POS and NEG", y=0.96, fontsize=14, fontweight='bold')
# ax.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
ax.set_xlabel("Index", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax.set_ylabel("RT difference", fontdict={'fontsize': 12, 'fontweight':'bold'})

# ax.set_xlim(0, right=ax.get_xlim()[1])
ax.set_ylim(top=0.05)

fig.savefig("RT_Difference.png", dpi=300)



fig, ax = plt.subplots(figsize=(10,6.18))
ax.grid(True, which='major', linestyle=':')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

x = data.index
y = data.RT_DIFF
ax.plot(x, y, 'o', color='skyblue')
ax.plot(x, y, '-ok', color='darkred')

for i, j in zip(x,y):
    
    if i == 8:
        label = data.iloc[i,0].split(' ')[-1]
        ax.annotate(label, (i,j-0.001), horizontalalignment='center', va='top', rotation=285, fontsize=10)
    
    elif i % 2 != 0:
        label = data.iloc[i,0].split(' ')[-1]
        ax.annotate(label, (i,j-0.001), horizontalalignment='center', va='top', rotation=285, fontsize=10)
    else:
        label = data.iloc[i,0].split(' ')[-1]
        ax.annotate(label, (i,j), horizontalalignment='center', va='bottom', rotation=285, fontsize=10)
    
# set titles and labels for x and y axis
fig.suptitle("Retention Time Differences (Absolute) between POS and NEG", y=0.96, fontsize=14, fontweight='bold')
# ax.set_title("(C>0.8, RT +/- 0.08min)", fontsize=12, fontweight='bold', fontstyle='italic')
ax.set_xlabel("Index", fontdict={'fontsize': 12, 'fontweight':'bold'})
ax.set_ylabel("RT difference", fontdict={'fontsize': 12, 'fontweight':'bold'})

# ax.set_xlim(0, right=ax.get_xlim()[1])
ax.set_ylim(bottom=-0.03)

fig.savefig("RT_Difference_absolute.png", dpi=300)



