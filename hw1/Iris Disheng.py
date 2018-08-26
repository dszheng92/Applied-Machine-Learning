
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

base_path = "/Users/disheng/Desktop/"
filename = "iris.data.txt"
path_to_file = os.path.join(base_path, filename)


# In[3]:

data = pd.read_csv(path_to_file, sep=",", header=None)
data.columns = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "class"]


# In[4]:

color = ['red' if l == "Iris-setosa" else 'green' if l == "Iris-virginica" else 'yellow' for l in data["class"]]
groups = ("Iris-setosa", "Iris-virginica", "versicolor") 


# In[5]:

plt.scatter(data["sepalLength"], data["sepalWidth"], c=color, label=groups)
plt.ylabel("Sepal Width")
plt.xlabel("Sepal Length")
#plt.legend(loc='upper left')




# In[6]:

plt.scatter(data["sepalLength"], data["petalLength"], c=color, label=groups)
plt.ylabel("Petal Width")
plt.xlabel("Sepal Length")


# In[7]:

plt.scatter(data["sepalLength"], data["petalWidth"], c=color, label=groups)
plt.xlabel("Sepal Length")
plt.ylabel("Septal Width")


# In[8]:

plt.scatter(data["sepalWidth"], data["petalLength"], c=color, label=groups)
plt.xlabel("Sepal Width")
plt.ylabel("Septal Length")


# In[9]:

plt.scatter(data["sepalWidth"], data["petalWidth"], c=color, label=groups)
plt.xlabel("Sepal Width")
plt.ylabel("Septal Width")


# In[10]:

plt.scatter(data["petalLength"], data["petalWidth"], c=color, label=groups)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

