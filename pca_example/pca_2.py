import numpy as np 
import scipy.ndimage
import scipy.misc
import sys 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import pandas as pd

A = 2 
N=150
k=2

df = pd.read_csv('iris.csv', names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


training = [] 
for item in x:
    a=np.zeros(shape=(A*A,1))
    a[1]=float(item[1])
    a[0]=float(item[0])
    a[2]=float(item[2])
    a[3]=float(item[3])
    training.append(a)


mean=np.zeros(shape=(A*A,1))

for t in training:
    mean+=t
    
mean=mean/N

X=np.zeros([A*A,N])
s=[]
for t in training:
   s.append(t-mean)
X=np.column_stack(s)

Q=X.T.dot(X)

l, V = np.linalg.eig(Q/(N-1))


v=[]
for i in range(k):
    pom=np.zeros([N,1])
    pom=V[:,i].reshape((N,1)) 
    v.append(pom)

 # If there are {\displaystyle n} n observations with {\displaystyle p} p variables, then the number of distinct principal components is min(n-1,p)

u=[]
for i in range(k):
    vektor=(np.dot(X, v[i])/np.sqrt(N*l[i]))
    u.append(vektor)
        
projected = [] 
for t in training:
    z = np.zeros([k,1])
    for i in range(k):
        z[i,0] = (u[i].T.dot(t-mean))[0,0]
    projected.append(z)


with open('eigenvectors.txt', 'w') as the_file:
    for u1 in u:
        for x in u1:
            the_file.write(str(x[0]))  
            the_file.write(' ')  
        the_file.write('\n')  

with open('projected.txt', 'w') as the_file:
    for u1 in projected:
        for x in u1:
            the_file.write(str(x[0]))  
            the_file.write(' ')  
        the_file.write('\n')  

with open('mean.txt', 'w') as the_file:
    for u1 in mean:
        for x in u1:
            the_file.write(str(x))  
            the_file.write(' ')  
        the_file.write('\n')  

x1=[]
y1=[]

for p in projected:
    i=0
    for m in p:
        if i%2==0:
            x1.append(m[0])
        elif i%2==1:
            y1.append(m[0])
        i+=1

col = ['blue', 'red', 'green']
for i in range(N):
    if y[i]=='setosa':
        cc = col[1]
    elif y[i] == 'versicolor':
        cc= col[2]
    else:
        cc=col[0]
    plt.scatter(x1[i],y1[i], c=cc,s=30)

plt.show()
