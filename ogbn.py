import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
import random
import json
import networkx as nx
import pylab as plt
import matplotlib.transforms as mtransforms
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
from scipy.stats import bootstrap

def bootstrapped(data,n_resamples = 50):
    data = (data,)
    bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95,n_resamples=n_resamples,
                         random_state=1, method='percentile')
    return bootstrap_ci.confidence_interval
color_2 = {
    'paper':'#bf5700'
}
## mapping fields of papers 
M={
 'computer science': 14
 }
MP = {M[i]:i for i in M}
CS = [14]## CS

FM = {}
for f in CS:
    FM[f] = 'CS'
FN = {
    'CS':'Computer Science'
}
# path ='/Cite/RemoteTeam_DataForMainFigrues/'
p_y = {}
Dis_mean = {}
TS = {}
p_d = {}
p_f = {}
with open('filtered_data.txt', 'r') as f:
    # 遍历文件的每一行
    for line in f:
        line = line.strip('\n').split('\t')  # 去除行尾换行符并按制表符分割成列表
        p = int(line[0])  # 提取论文ID并转换为整数
        p_y[p] = int(line[1])  # 存储论文ID和年份的映射关系
        Dis_mean[p] = float(line[4])  # 存储论文ID和距离均值的映射关系
        TS[p] = int(line[3])  # 存储论文ID和团队规模的映射关系
        if line[-1] != 'nan':  # 如果最后一列不是'nan'
            p_d[p] = float(line[-1])  # 存储论文ID和分数的映射关系
        if line[2] != '-1':  # 如果第三列不是'-1'
            p_f[p] = M[line[2]]  # 存储论文ID和领域的映射关系
# # 打开原始文件
# with open(path + 'Paperid_Year_Discipline_Teamsize_Distance_Dscore.txt', 'r') as f:
#     # 遍历文件的每一行
#     for line in f:
#         count += 1  # 统计行数
#         line = line.strip('\n').split('\t')  # 去除行尾换行符并按制表符分割成列表
#         p = int(line[0])  # 提取论文ID并转换为整数
#         p_y[p] = int(line[1])  # 存储论文ID和年份的映射关系
#         Dis_mean[p] = float(line[4])  # 存储论文ID和距离均值的映射关系
#         TS[p] = int(line[3])  # 存储论文ID和团队规模的映射关系
#         if line[-1] != 'nan':  # 如果最后一列不是'nan'
#             p_d[p] = float(line[-1])  # 存储论文ID和分数的映射关系
#         if line[2] != '-1':  # 如果第三列不是'-1'
#             p_f[p] = M[line[2]]  # 存储论文ID和领域的映射关系

# 筛选第三列为"computer science"的行，并将其写入新的文本文件
# with open('filtered_data.txt', 'w') as filtered_file:
#     with open(path + 'Paperid_Year_Discipline_Teamsize_Distance_Dscore.txt', 'r') as f:
#         for line in f:
#             line = line.strip().split('\t')  # 去除行首尾空格并按制表符分割成列表
#             if line[2] == 'computer science':  # 如果第三列为"computer science"
#                 filtered_file.write('\t'.join(line) + '\n')  # 将该行写入新文件
#                 count+=1

# 年代图
# Y_p10 = defaultdict(lambda :defaultdict(list))
# for p in p_f:
#     if p in Dis_mean:
#         Y_p10[FM[p_f[p]]][p_y[p]].append(Dis_mean[p])
# Y_p11 = {}
# for f in Y_p10:
#     a = {y:np.mean(Y_p10[f][y]) for y in Y_p10[f]} # (paper)field-year: average collaboration distance 
#     Y_p11[f] = a
# fig,ax=plt.subplots(figsize=[6, 6])

# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# markers_pa = ["v","o","^"]
# n = 0
# for f in ['CS']:
#     ys = [y for y in range(1900,2021) if y in Y_p11[f]]
#     rp = [Y_p11[f][y] for y in ys]
#     ax.scatter(ys,rp,fc = color_2['paper'],edgecolors='none',alpha=0.1,s=40,marker=markers_pa[n])
#     ys = list(map(int,list(np.convolve(ys, np.ones(16)/16, mode='valid') )))
#     rp = np.convolve(rp, np.ones(16)/16, mode='valid')
#     ax.plot(ys,np.array(rp),c = color_2['paper'],lw = 2)
#     ax.scatter(ys[-1],np.array(rp)[-1],fc = color_2['paper'],edgecolors='none',marker=markers_pa[n],\
#                 s=200,label = f)
#     n +=1

# ax.set_xlabel('Year',size = 16)

# ax.tick_params(axis='x',labelsize=14)
# ax.tick_params(axis='y',labelsize=14)
# ax.set_ylabel('Collaboration distance (km)',size=16)
# trans = mtransforms.ScaledTranslation(-60/72, 15/72, fig.dpi_scale_trans)
# ax.text(.0, 1.0, '', transform=ax.transAxes + trans,
#             fontsize=30, va='bottom')
# ax.set_xlim([1960,2021])
# ax.set_ylim([100,1400])
# ax.legend(frameon=False,fontsize=12,loc=4)

# plt.tight_layout()
# plt.show()
## D-score vs. collaboration distance
# bin size = 200 km
Dbins = defaultdict(list)
for p in p_d:
    if Dis_mean[p] <= 0:
        d = 0
    if Dis_mean[p] > 0 and Dis_mean[p]<=600:
        d = int(Dis_mean[p]/200+1)*200
    if Dis_mean[p]>600:
        d = 10000
    Dbins[d].append(p_d[p])

blp = np.sum([int(p_d[p]>0) for p in p_d])/len(p_d)

## threshold of being disruptive : D-score > 0
thdpa = 0


x0 = [i for i in sorted(list(Dbins.keys()))]
y0 = [len(np.array(Dbins[i])[np.array(Dbins[i])>thdpa])/len(Dbins[i]) for i in x0 if len(Dbins[i])>0]
y0ci = [bootstrapped(list(map(lambda x:int(x>0), Dbins[i])),100) for i in x0 if len(Dbins[i])>0]

fig,ax2=plt.subplots(1, 1, figsize=[6, 6])

## Main figure
ax2.plot(list(range(len(x0))),np.array(y0)/blp,'-',c = color_2['paper'],lw = 8,label='Papers',zorder=100)


for i in range(len(x0)):
    y0ci_ = np.array(y0ci)/blp
    ax2.plot([i,i],[y0ci_[i][0],y0ci_[i][1]],lw=2,color=color_2['paper'],zorder=10)
    ax2.plot([i-0.1,i+0.1],[y0ci_[i][0],y0ci_[i][0]],lw=2,color=color_2['paper'],zorder=10)
    ax2.plot([i-0.1,i+0.1],[y0ci_[i][1],y0ci_[i][1]],lw=2,color=color_2['paper'],zorder=10)


ax2.axhline(1,ls='--',color='grey',lw=3)
ax2.tick_params(axis='x',labelsize=14)
ax2.tick_params(axis='y',labelsize=14)
ax2.set_yticks([0.8,0.9,1,1.1])
ax2.set_yticklabels(tuple(['0.80','0.90','1.00','1.10']))
ax2.set_xticks([0,1,2,3,4])
ax2.set_xticklabels(tuple(['0','200','400','600','600+']))
ax2.set_xlabel('Collaboration distance (km)',size = 16)
ax2.set_ylabel('Relative probability of disruption',size = 16)
ax2.legend(fontsize=14,frameon=False,loc=1)
trans = mtransforms.ScaledTranslation(-60/72, 15/72, fig.dpi_scale_trans)
ax2.text(.0, 1.0, '', transform=ax2.transAxes + trans,
            fontsize=25, va='bottom', fontfamily='arial')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.show()
