import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Downlaod the data
dataframe_all = pd.read_csv('pml-training.csv', low_memory=False)
num_rows = dataframe_all.shape[0]

# Clean the data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan == 0]
#remove the colums with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]
#remove the first 7 colums which contain no discriminative information
dataframe_all = dataframe_all.ix[:,7:]

#Creature feature vectors
x = dataframe_all.ix[:, :-1].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(x)

# t distributed stochastic neighbour embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state = 0)
x_test_2D = tsne.fit_transform(x_std)

# scatter plot the sample points among 5 classes
markers = ('s', 'd', 'o', '^', 'v')
color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan'}
plt.figure()

for idx, cl in enumerate(np.unique(x_test_2D)):
    plt.scatter(
        x = x_test_2D[cl, 0],
        y = x_test_2D[cl, 1],
        c = color_map[idx],
        marker = markers[idx],
        label = cl
    )

plt.show()
