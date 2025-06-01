import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, NMF
import os
import matplotlib
matplotlib.use('TkAgg')

#1.1.  Read in data and explore
df = pd.read_csv('../data_number_nine.csv', header='infer')
df.shape

df.head(5)

#1.2. Visualize the data
# Define a suitable visualization function based on imshow().
# Visualizes the whole dataset at once as pixel image.
def ShowMe(X):
    Y= 1.0 - X
    plt.imshow(Y, cmap='gray')
    plt.show()
X = np.array(df)
ShowMe(X)

#1.3. Visualize the reduced dimensional input by PCA
# Define a function that returns reduced dimensional input.
def reducedInputPCA(X,nPC):
    pca = PCA(n_components = nPC)                           # Define a PCA object for a given number of target PCs.
    X_pca = pca.fit_transform(X)                            # Get the transformed scores.
    return pca.inverse_transform(X_pca)                     # Bring back the transformed scores to the original coordinate system.
# Visualize the reduced dimensional input for different cases.
# As we shrink the dimension, the image gets less clear.
for nPC in [23, 10, 5, 3, 1]:
    Z = reducedInputPCA(X,nPC)
    print( "N# of PCs = " + str(nPC))
    ShowMe(Z)
