#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @date: 2020/9/24 21:27
# @author：yukyin
# Talk is cheap show me the code!

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
pred=pd.read_csv('SQ-portion1.csv',index_col=0)
# print(pred)
row_name=pred._stat_axis.values.tolist()  # 行名称
col_name=pred.columns.values.tolist()  # 列名称
print(row_name)
# print(col_name)
row_name_DQA=[i for i in row_name if 'DQAs' in i]
row_name_SQA=[i for i in row_name if 'SQAs' in i]
row_name_EQA=[i for i in row_name if 'EQAs' in i]
col_name_adv=[i for i in col_name if '-adv' in i]
col_name_mot=[i for i in col_name if '-mot' in i]
col_name_adv_name=[i.rsplit('-',1)[0] for i in col_name_adv]
col_name_mot_name=[i.rsplit('-',1)[0] for i in col_name_mot]
print(row_name_DQA)
print(row_name_SQA)
print(row_name_EQA)

row_name_=[row_name_DQA,row_name_SQA,row_name_EQA]
col_name_=[col_name_adv_name,col_name_mot_name]


n=0#'Adv','Mot'
m=2#'DQAs','EQAs','SQAs'
# t=2#DQAs-SQAs-EQAs (SQAs or EQAs:1 or 2)


cat=['DQAs','EQAs','SQAs']
adv_mot=[col_name_adv,col_name_mot]
adv_mot_name=['Adv','Mot']

y1 = pred[adv_mot[n][0]][pred.index.str.contains(cat[m])].values
y2=pred[adv_mot[n][1]][pred.index.str.contains(cat[m])].values
y3=pred[adv_mot[n][2]][pred.index.str.contains(cat[m])].values
y4=pred[adv_mot[n][3]][pred.index.str.contains(cat[m])].values
y5 = pred[adv_mot[n][4]][pred.index.str.contains(cat[m])].values
y6=pred[adv_mot[n][5]][pred.index.str.contains(cat[m])].values
y7=pred[adv_mot[n][6]][pred.index.str.contains(cat[m])].values
y8=pred[adv_mot[n][7]][pred.index.str.contains(cat[m])].values
y9=pred[adv_mot[n][8]][pred.index.str.contains(cat[m])].values


plt.figure(figsize=(3.8,2.8))

# x=np.array([i.split('-PQAs')[0] for i in row_name_[m]])
x=np.array([i for i in row_name_[m]])
y1=np.array(y1)
y2=np.array(y2)
y3=np.array(y3)
y4=np.array(y4)
y5=np.array(y5)
y6=np.array(y6)
y7=np.array(y7)
y8=np.array(y8)
y9=np.array(y9)

# print(x)
# print(y2)
#
# exit()
color=['#927076',
'#301E1D',
'#BC6D43',
'#D39F72',
'#9CA1A6',
'#87A0C6',
# '#927076',
# '#301E1D',
# '#BC6D43',
'#7F3548',
'#79905E',
'#205F95'
       ]




# plt.xlabel("Portion",fontsize=8) # x轴名称
plt.ylabel(cat[m]+'-'+adv_mot_name[n]+"-accuracy",fontsize=9) # y 轴名称
plt.xticks(rotation=40)
if not np.isnan(y1).any():
    plt.plot(x, y1,color=color[0],marker='o',ms=3,label=col_name_[0][0])
if not np.isnan(y2).any():
    plt.plot(x, y2, color=color[1],marker='*', ms=3,label=col_name_[0][1])
if not np.isnan(y3).any():
    plt.plot(x, y3,color=color[2],marker='v',ms=3,label=col_name_[0][2])
if not np.isnan(y4).any():
    plt.plot(x, y4, color=color[3],marker='^', ms=3,label=col_name_[0][3])
if not np.isnan(y5).any():
    plt.plot(x, y5,color=color[4],marker='<',ms=3,label=col_name_[0][4])
if not np.isnan(y6).any():
    plt.plot(x, y6, color=color[5],marker='>', ms=3,label=col_name_[0][5])
if not np.isnan(y7).any():
    plt.plot(x, y7, color=color[6],marker='D', ms=3,label=col_name_[0][6])
if not np.isnan(y8).any():
    plt.plot(x, y8,color=color[7],marker='s',ms=3,label=col_name_[0][7])
if not np.isnan(y9).any():
    plt.plot(x, y9, color=color[8],marker='p', ms=3,label=col_name_[0][8])

    # filled_markers = (
    #     'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
    #     'P', 'X')

plt.axhline(y=y1[0], color=color[0],linestyle="dashed")
plt.axhline(y=y2[0], color=color[1],linestyle="dashed")
plt.axhline(y=y3[0], color=color[2],linestyle="dashed")
plt.axhline(y=y4[0], color=color[3],linestyle="dashed")
plt.axhline(y=y5[0], color=color[4],linestyle="dashed")
plt.axhline(y=y6[0], color=color[5],linestyle="dashed")
plt.axhline(y=y7[0], color=color[6],linestyle="dashed")
plt.axhline(y=y8[0], color=color[7],linestyle="dashed")
plt.axhline(y=y9[0], color=color[8],linestyle="dashed")



plt.xticks(x,fontsize=6.8)
plt.yticks(fontsize=7.5)

# plt.yticks([94,96])#squad1.1-F1
if n==0 and m==2:
    plt.legend(loc='lower left',fontsize=8)
    leg = plt.legend(loc='best',fontsize=6.5)
# for lh in leg.legendHandles:
#     lh._legmarker.set_alpha(1)
# frame = legend.get_frame()
# frame.set_alpha(1)
# frame.set_facecolor('none') # 设置图例legend背景透明
plt.savefig('model-'+adv_mot_name[n]+'-'+cat[m]+'.png',dpi=1000,bbox_inches = 'tight')
plt.show()

