import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler as SS

###############
## NOTE: BAD ##
###############

# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column 
# being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is 
# decreasing, and  eigenvectors have  same order as their associated eigenvals
def my_pca(data):
    data_mean = np.mean(data.T, axis=1)
    x, y = data[:,0].T, data[:,1].T
    # compute eigen decomposition of covar matrix
    evals, evecs = np.linalg.eig(np.cov(x, y))   # check if eig vs eigh
    # return in descending order
    evals = evals[::-1]
    evecs = evecs[:,::-1]
    #print('mean', data_mean)
    return evals, evecs, data_mean

def chinese_pca(data):
    # centering 
    average = np.mean(data, axis=0)
    print('chinese average', average)
    mid = data - average
    cov = np.cov(mid.T) 
    
    # eigenvalue & eigenvector
    e_value, e_vector = np.linalg.eig(cov)
    
    # sort
    s = np.argsort(e_value)[::-1] # reverse
    e_val = e_value[s]
    e_vec = e_vector[:, s]
        
    return e_val, e_vec

def plot_original(x, y):
    plt.scatter(x, y)
    plt.axis('equal')
    plt.show()

def plot_original_classes(x, y, classes):
    plt.figure()
    labels = {0:'Weed', 1:'Crop'}      # weed is 0, crop is 1
    colors = {0:'r', 1:'g'}
    for l in np.unique(classes):
        ix = np.where(classes==l)
        plt.scatter(x[ix], y[ix], c=colors[l],label=labels[l])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Top-2 Feature Weed Data')
    plt.legend()
    plt.show()

def plot_PC(x, y, evals, evecs, dmean):
    # plot original data as scatter
    plt.scatter(x-6.93, y-20.57)
    plt.axis('equal')

    # Compute the corresponding standard deviations
    s0 = np.sqrt(evals[0])
    s1 = np.sqrt(evals[1])

    #plt.plot([6.93, s0*evecs[0,0]], [20.57, s0*evecs[1,0]], 'r')
    #plt.plot([6.93, s1*evecs[0,1]], [20.57, s1*evecs[1,1]], 'r')
    plt.plot([0, s0*evecs[0,0]], [0, s0*evecs[1,0]], 'r')
    plt.plot([0, s1*evecs[0,1]], [0, s1*evecs[1,1]], 'r')

    plt.title('Plot of Murder Data with PC Eigenvectors Pointing Out of Mean')
    plt.xlabel('Component 1')   # TODO might be incorrect label
    plt.ylabel('Component 2')   # TODO ^
    plt.show()

def plot_scaling(data, num_dim, classes):
    pca = PCA(n_components=num_dim)
    scaled_data = pca.fit_transform(data)
    #plot_original(scaled_data[:,0], data[:,1])

    plt.figure()
    X_data = -scaled_data[:,0]
    print('X_DATA', X_data.shape)
    Y_data = -scaled_data[:,1]
    labels = {0:'Weed', 1:'Crop'}      # weed is 0, crop is 1
    colors = {0:'r', 1:'g'}
    for l in np.unique(classes):
        ix = np.where(classes==l)
        plt.scatter(X_data[ix], Y_data[ix], c=colors[l],label=labels[l])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Pesticide Data Scaled Onto 2 Components')
    plt.legend()
    plt.show()



# Function: plot_variance_PC
# Plot of variance versus PC index (variance stabilizes)
# Input:    data features
# Output:   plot
# BAD TODO TODO TODO
def plot_variance_PC(data):
    #plt.plot(evals)
    pca = PCA().fit(data)
    evals = pca.explained_variance_ratio_
    plt.plot(evals)
    plt.title('Plot of Pesticide Data for Principal Components vs Variance')
    plt.xlabel('# of Principal Components')
    plt.ylabel('Projected Variance')
    plt.show()

# Function: plot_variance_PC
# Plot of cumulative variance versus PC index
# Input:    data features
# Output:   plot
def plot_variance_cumul_PC(data):
    pca = PCA().fit(data)
    evals = pca.explained_variance_ratio_
    plt.plot(np.cumsum(evals/np.sum(evals)))
    plt.xlabel('# of Principal Components')
    plt.ylabel('Cumulative projected variance')
    plt.show()



# Main method
np.set_printoptions(precision=2, suppress=True)
# weed/crop data
weed_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
weed_X = weed_train[:, :-1]                 # (1000, 13)
weed_y = weed_train[:, -1]
# murder data
murder_X = np.loadtxt('murderdata2d.txt')   # (20, 2)

# Perform PCA on murder & pestilence data sets
print('\n============EXERCISE 1============')
print('Performing PCA on the murder dataset . . .')
evals, evecs, dmean = my_pca(murder_X)
### Faster version of PCA:
###f_evals, f_evecs = np.linalg.eig(np.cov(murder_X[:,0], murder_X[:,1]))
print('\nUnit vectors spanning principal components: ')
print(evecs)
print('\nVariance of each of the above components: ')
print(evals)

#============================================
## TRY EIGENVAL VECT ON PESTILENCE
iris_data = datasets.load_iris()
iris_X = iris_data.data
ci_evals, ci_evecs = chinese_pca(iris_X)
print('\niris chinese PCA eigenvectors:')
print(ci_evecs[0:2,])
print('\niris chinese PCA eigenvalues:')
print(ci_evals)

iris_X_scaled = SS().fit_transform(iris_X)
pca_iris = PCA()
pca_iris.fit(iris_X_scaled)
si_evecs = pca_iris.components_
si_evals = pca_iris.explained_variance_
idx = si_evals.argsort()[::-1]
si_evals = si_evals[idx]
si_evecs = si_evecs[:,idx]
print('\niris sklearn PCA eigenvectors:')
print(si_evecs[0:2,])
print('\niris sklearn PCA eigenvalues:')
print(si_evals)

cw_evals, cw_evecs = chinese_pca(weed_X)
print('\nweed chinese PCA eigenvectors:')
print(cw_evecs[0:2,])
print('\nweed chinese PCA eigenvalues:')
print(cw_evals)

pca_weed = PCA()
sw_evecs = pca_weed.fit_transform(weed_X)
sw_evals = pca_weed.explained_variance_

print('\nweed sklearn PCA eigenvectors:')
print(sw_evecs)
print('\nweed sklearn PCA eigenvalues:')
print(sw_evals)


#=======================================================
print('\nPlotting scatterplot of murder dataset with mean and eigenvectors\n(scaled by standard deviation) pointing out of the mean . . .')
plot_PC(murder_X[:,0], murder_X[:,1], evals, evecs, dmean)

print('\nPlotting graph of pestilence dataset with principal components vs variance')
#oof_evals, oof_evecs = np.linalg.eig(np.cov(weed_X))
#plt.plot(oof_evals)
plt.plot(cw_evals)
plt.title('chinese')
plt.show()
plt.plot(sw_evals)
plt.title('sklearn')
plt.show()


# compare against built in model
#pca = PCA()
#pca.fit(murder_X)
#built_in_evecs = pca.components_
#built_in_evals = pca.explained_variance_
#print('BUILT IN principal components: ', pca.components_)
#print('BUILD IN Variance of components: ', pca.explained_variance_)

#idx = built_in_evals.argsort()[::-1]
#built_in_evals = built_in_evals[::-1]
#built_in_evecs = built_in_evecs[:,::-1]


#plot_PC(murder_X[:,0], murder_X[:,1], built_in_evals[idx], built_in_evecs[:,idx], dmean)

#print(fake_evals)
#plt.title('Fake evals')
#plt.plot(fake_evals)
#plot_variance_PC(weed_X)
##plot_variance_cumul_PC(weed_X)
plot_scaling(weed_X, 2, weed_y)
##print(dmean)