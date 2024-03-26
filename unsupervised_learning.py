# https://www.kaggle.com/code/umutsefkansak/unsupervised-learning?rvi=1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%

data = pd.read_csv("Iris.csv")

data.info()

data.isnull().sum()

print("Total Null values",data.isna().sum().sum())

#%%
sns.pairplot(data,hue="Species")

#%%

df1 = data.iloc[:,[3,4]]

#%%

plt.scatter(df1.PetalLengthCm,df1.PetalWidthCm)
plt.show()

#%%

from sklearn.cluster import KMeans

wcss = []

for i in range(1,15):
    kmeans2 = KMeans(n_clusters=i)
    kmeans2.fit(df1)
    wcss.append(kmeans2.inertia_)




plt.plot(range(1,15),wcss)
plt.xlabel("Number of K")
plt.ylabel("Wcss")
plt.show()


#%% we can choose k=3 -> elbow method

kmeans = KMeans(n_clusters=3)
clusters_kmean = kmeans.fit_predict(df1)

df1["label"] = clusters_kmean

#%%
plt.scatter(df1.PetalLengthCm[df1.label == 0],df1.PetalWidthCm[df1.label == 0],color = "red")
plt.scatter(df1.PetalLengthCm[df1.label == 1],df1.PetalWidthCm[df1.label == 1],color = "blue")
plt.scatter(df1.PetalLengthCm[df1.label == 2],df1.PetalWidthCm[df1.label == 2],color = "green")
plt.show()

#%%
from scipy.cluster.hierarchy import linkage,dendrogram

df2 = data.iloc[:,[3,4]]
merg = linkage(df2,method="ward")
dendrogram(merg)
plt.show()
#%% we can choose n_clusters = 3 -> dendrogram

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")

clusters_hierarchical = ac.fit_predict(df2)

df2["label"] = clusters_hierarchical

plt.scatter(df2.PetalLengthCm[df2.label == 0],df2.PetalWidthCm[df2.label == 0],color = "red")
plt.scatter(df2.PetalLengthCm[df2.label == 1],df2.PetalWidthCm[df2.label == 1],color = "blue")
plt.scatter(df2.PetalLengthCm[df2.label == 2],df2.PetalWidthCm[df2.label == 2],color = "green")
plt.show()
 


