# PCA Visual Demo
# Programmmed by Olac Fuentes
# Last modified November 3, 2020

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

#Generate some 2D data
t =  np.arange(-500,500,4)
x1 =  20*np.sin(t/4)
x2 =  (40*np.random.random(size = len(x1))-.5)

x = x1+x2
alpha = np.pi/6
rot = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
p = np.matmul(rot,np.vstack((t,x))) + np.array([[300],[-100]])

fig, ax = plt.subplots()
ax.plot(p[0],p[1],'.', markersize=1)
ax.set_title('Original Data')
ax.set_aspect(1.0)
plt.grid(True)

# Perform PCA projection
pca = PCA()
pca.fit(p.T)
print('Explained variance:',pca.explained_variance_ratio_)
print('Data mean:',pca.mean_)
print('First component :',pca.components_[0]) 
print('Second component :',pca.components_[1]) 


fig, ax = plt.subplots()
ax.set_title('Principal Components')
ax.plot(p[0],p[1],'.', markersize=1)
for i in range(2):
    p0 = -400*pca.components_[i] + pca.mean_
    p1 =  400*pca.components_[i] + pca.mean_
    ax.plot([p0[0],p1[0]],[p0[1],p1[1]])
ax.set_aspect(1.0)
plt.grid(True)
fig, ax = plt.subplots()

pt = pca.transform(p.T).T
ax.plot(pt[0],pt[1],'.', markersize=1)
ax.set_aspect(1.0)
ax.set_title('Data Projected to Principal Components')
plt.grid(True)

pca = PCA(n_components=1)
pca.fit(p.T)
pt = pca.transform(p.T)

pit = pca.inverse_transform(pt).T

fig, ax = plt.subplots()
ax.plot(pit[0],pit[1],'.', markersize=1)
plt.grid(True)
ax.set_title('Inverse projection of data Projected to 1 principal component')

