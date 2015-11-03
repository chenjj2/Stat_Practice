'''
cluster on test_marcy_cluster.py
'''

### import 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


### data
para_chain = np.loadtxt('test_marcy_cluster_parameter.out')
n_step, total_para = np.shape(para_chain)
para_chain = para_chain.reshape(5,n_step,3)

alpha = np.hstack(para_chain[:,n_step/2:,0])
beta = np.hstack(para_chain[:,n_step/2:,1])


### 2d histogram
xedge = np.linspace(0.,1.,21)
yedge = np.linspace(0.,1.,21)

H, xedges, yedges = np.histogram2d(beta,alpha,bins=100)

#print H

### plot
plt.imshow(np.log(H+1), origin='low', cmap='Greys')
plt.colorbar()
plt.xlabel(r'$\alpha$'); plt.ylabel(r'$\beta$')
plt.savefig('Figure/test_marcy_cluster_2dhist.png')


'''
### scikit cluster
# http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#example-cluster-plot-mini-batch-kmeans-py
data = np.vstack((alpha,beta)).transpose()
n_clusters=3

k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init = 10)
k_means.fit(data)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

print k_means_cluster_centers
'''

print 'end'


