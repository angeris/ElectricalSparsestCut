import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

df = pd.read_csv('higher_k_cut_ratio_values_withtime_morerandomcuts.csv', index_col = False)
df['RatioTime'] = df['elec_time']/df['ckr_time']
result = df.pivot_table(index='n', columns='k', values='ratioCKRElectrical')
time_ckr = df.pivot_table(index='n', columns='k', values='ckr_time')
time_elec = df.pivot_table(index='n', columns='k', values='elec_time')
time_ratio = df.pivot_table(index='n', columns='k', values='RatioTime', aggfunc = np.mean)
# plt.plot(time_ckr.loc[:,4])
# plt.plot(time_elec.loc[:,4])
sns.heatmap(time_ratio, annot=False, fmt="g", cmap='viridis', yticklabels = 20)#, norm=LogNorm(vmin=.5, vmax=10))

plt.show()
print(result.head)
sns.heatmap(result, annot=False, fmt="g", cmap='viridis', yticklabels = 20)#, norm=LogNorm(vmin=.5, vmax=10))
plt.show()
# print (df.head)
# df.set_index(['n', 'k']).reindex.ratioCKRElectrical.unstack(0).pipe(plt.imshow)

# plt.pcolor(df[['n','k','ratioCKRElectrical']])
# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
# plt.show()
