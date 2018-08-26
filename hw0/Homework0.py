
# Disheng Zheng
# Dan Terry

# In[133]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[134]:

dataframe = pd.read_csv('iris.data', header=None, names=['sep_len', 'sep_wid', 'pet_len', 'pet_wid', 'species'], dtype={'species': 'category'})


# In[135]:

dataframe['species'].describe()


# In[136]:

cmap = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'yellow'}
scatter_matrix(dataframe.iloc[ : ,0:4], alpha=0.2, figsize=(36, 36), diagonal='kde', c=[cmap.get(c) for c in dataframe.species], s=1000)

