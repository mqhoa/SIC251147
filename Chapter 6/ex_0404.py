import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, NMF
import os
import matplotlib
matplotlib.use('TkAgg')

#1.1.  Read in data and explore
df = pd.read_csv('data_number_nine.csv', header='infer')
df.shape

df.head(5)
