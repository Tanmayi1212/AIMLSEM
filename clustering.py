import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

blobs,_=make_blobs(n_samples=300,centers=4,random_state=42)
moons,_=make_moons(n_samples=300,noise=0.05,random_state=42)

scaler=StandardScaler()
blobs=scaler.fit_transform(blobs)
moons=scaler.fit_transform(moons)

kmeans=KMeans(n_clusters=4,random_state=42)
agglo=AgglomerativeClustering(n_clusters=4)
dbs=DBSCAN(eps=0.3,min_samples=4)

kmeans_blobs=kmeans.fit_transform(blobs)
agglo_blobs=agglo.fit_transform(blobs)
dbs_blobs=dbs.fit_transform(blobs)

kmeans_moons=kmeans.fit_transform(moons)
agglo_moons=agglo.fit_transform(moons)
dbs_moons=dbs.fit_transform(moons)

fig,ax=plt.subplots(3,2,figsize=(12,12))
ax[0,0].scatter(blobs[:,0],blobs[:,1],c=kmeans_blobs,cmap='viridis')
ax[0,0].set_title("KMeans on blobs")

ax[1,0].scatter(blobs[:,0],blobs[:,1],c=agglo_blobs,cmap='viridis')
ax[1,0].set_title("KMeans on blobs")

ax[2,0].scatter(blobs[:,0],blobs[:,1],c=dbs_blobs,cmap='viridis')
ax[2,0].set_title("KMeans on blobs")

ax[0,1].scatter(blobs[:,0],blobs[:,1],c=kmeans_moons,cmap='viridis')
ax[0,1].set_title("KMeans on blobs")

ax[0,2].scatter(blobs[:,0],blobs[:,1],c=agglo_moons,cmap='viridis')
ax[0,2].set_title("KMeans on blobs")

ax[0,3].scatter(blobs[:,0],blobs[:,1],c=dbs_moons,cmap='viridis')
ax[0,3].set_title("KMeans on blobs")

plt.tight_layout()
plt.show()