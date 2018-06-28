import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

data=pd.read_csv('./food-choices/food_coded.csv')

data = data[data['weight'].apply(lambda x: str(x).isdigit())]
Y = data[['Gender', 'eating_out']]
X = data[['weight']]

n = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in n]
	
score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]

plt.plot(n, score)
plt.xlabel('Broj klastera')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# PCA algoritam se koristi da bi 
# se podaci koji su mozda
# previse rasprseni
# pretvorili u linearne kombinacije
# i na taj nacin bili lakse citljivi
pca = PCA(n_components=2).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)

# Algoritam K-sredina
# pravi se 3 klastera 
# sto smo zakljucili eksperimaentalno
kmeans = KMeans(n_clusters=3)
kmeansoutput = kmeans.fit(Y)

# mogu se pogledati vrednosti
# krajnjih centroida
print(kmeans.cluster_centers_)

# klasteri se prikaziuju
plt.figure('8 klastera K-Means')
plt.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
plt.xlabel('Tezina')
plt.ylabel('Konzumiranje prekomernih kolicina hrane i pol')
plt.title('3 klastera K-Means')
plt.show()


#f1 = data['Gender'].values
#f2 = data['eating_out'].values
#X = np.array(list(zip(f1, f2)))
#plt.scatter(f1, f2, c='black', s=7)

#kmeans = KMeans(n_clusters=3)
#kmeans = kmeans.fit(X)
#labels = kmeans.predict(X)
#centroids = kmeans.cluster_centers_
#print(centroids)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X, X, X)
#x.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

#data=data.iloc[:,0:10]
#print(data)

#data.Gender = data.Gender.astype(int)
#data = data[data['Gender'].apply(lambda x: str(x).isdigit())]
#data.reset_index(drop=True, inplace=True)

#ss = StandardScaler()
#data=ss.fit_transform(data)