#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# import data
#data = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")# input data
data = pd.read_csv("datasets\\iris.csv")

# input data
df = data[["sepal_length", "sepal_width"]]

#data distribution
v1 =df["sepal_length"]
plt.hist(v1,color='g')
plt.xlabel('value')
plt.ylabel('count')
plt.title('sepal_length')
plt.show()
v2 =df["sepal_width"]
plt.hist(v2,color='r')
plt.xlabel('value')
plt.ylabel('count')
plt.title('sepal_width')
plt.show()

# check nan value
if v1.isnull().values.any() == False:
    print("art_daily_small_noise doesn't have Nan value.")
else:
    print("please check art_daily_small_noise Value.There are"+ v1.isnull().values.sum()+ "value.")
    
if v2.isnull().values.any() == False:
    print("art_daily_jumpsup doesn't have Nan value.")
else:
    print("please check art_daily_jumpsup Value There are"+ v2.isnull().values.sum()+ "value.")
    
# scatterplot of inputs data
plt.scatter(df["sepal_length"], df["sepal_width"])
plt.show()

# create arrays
X = df.values

# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)# fit model
print(nbrs.fit(X))

# distances and indexes of k-neaighbors from model outputs
# plot mean of k-distances of each observation
distances, indexes = nbrs.kneighbors(X)
plt.plot(distances.mean(axis =1))
plt.show()
# visually determine cutoff values > 0.15
outlier_index = np.where(distances.mean(axis = 1) > 0.15)
print(outlier_index)

# filter outlier values
outlier_values = df.iloc[outlier_index]
print(outlier_values)

# plot data
plt.scatter(df["sepal_length"], df["sepal_width"], color = "b", s = 65)# plot outlier values
plt.scatter(outlier_values["sepal_length"], outlier_values["sepal_width"], color = "r")
plt.show()

