#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @date: 2021/9/13 20:39 
# @author：yukyin
# Talk is cheap show me the code!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=['SQ','WQ']
# adv_mot_name=['Adv','Mot']
n=0

pred=pd.read_csv(dataset[n]+'-posneg1.csv',index_col=0)


# print(pred)
row_name=pred._stat_axis.values.tolist()  # 行名称
col_name=pred.columns.values.tolist()  # 列名称
# print(row_name)
# print(col_name)

color=[
'#7F3548',
'#BA849E',
'#C3A18F',
'#DDEAF0',
'#8E9894',
'#27302D',
'#6DADC8',
'#A9D5DE',
'#DBBFB2',
'#BB9B9E',
'#6B2F15',
'#040907',
'#6194A7',
'#7BA1C7',
'#5F8CB9',
'#A6CFE4',
    '#AACDD3'
    ]





y_pos_adv = tuple(pred.loc['pos-adv'].values)
y_neg_adv = tuple(pred.loc['neg-adv'].values)
y_pos_mot = tuple(pred.loc['pos-mot'].values)
y_neg_mot = tuple(pred.loc['neg-mot'].values)

ind = np.arange(0, 2*len(col_name), 2)  # the x locations for the groups
width = 0.2 # the width of the bars



fig, ax = plt.subplots(figsize=(4, 2.1))
rects1 = ax.bar(ind + width, y_pos_adv, width, color=color[1], align='edge', label='Adv-Pos')
rects2 = ax.bar(ind + 2 * width, y_pos_mot, width, color=color[2], align='edge', label='Mot-Pos')
rects3 = ax.bar(ind + 3*width, y_neg_adv, width, color=color[-1], align='edge', label='Adv-Neg')
rects4 = ax.bar(ind + 4 * width, y_neg_mot, width, color=color[4], align='edge', label='Mot-Neg')



# ax.set_title('Scores by group and gender')

plt.xticks(ind, col_name,rotation=60,fontsize=5.5)
plt.yticks(fontsize=5.5)
# plt.xlabel("Model",fontsize=5.5) # x轴名称
plt.ylabel(dataset[n]+"-Accuracy",fontsize=7.5) # y 轴名称
if n==1:
    ax.legend(loc='upper left',fontsize=6.5)


plt.savefig(dataset[n]+'-posneg.png',dpi=1000,bbox_inches = 'tight')
plt.show()
