import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data: pd.DataFrame = pd.read_csv('data/iris.csv')

data.boxplot(figsize=(10,6))
plt.show()


corr_mtx = data.corr()
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
plt.title('Correlation analysis')
plt.show()
