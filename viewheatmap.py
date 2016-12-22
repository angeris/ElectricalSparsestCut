import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('higher_k_cut_ratio_values.csv', index_col = False)
result = df.pivot_table(index='n', columns='k', values='ratioCKRElectrical')
print(result.head)
sns.heatmap(result, annot=False, fmt="g", cmap='viridis')
plt.show()
# print (df.head)
# df.set_index(['n', 'k']).reindex.ratioCKRElectrical.unstack(0).pipe(plt.imshow)

# plt.pcolor(df[['n','k','ratioCKRElectrical']])
# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
# plt.show()
