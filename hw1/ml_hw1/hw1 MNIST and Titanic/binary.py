import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
import itertools as it
import hw1_modules
import sklearn.metrics as skm
import matplotlib.pyplot as mp

data = pd.read_csv("train.csv", dtype={"label" : "category"})

#Dictionary of desired lables (0 and 1)
values = {"label" : ["0","1"]}

# Subset of data where labels is 0 or 1
data_01 = data[data.isin(values).any(1)]

#calculate pairwise distance between every element in dataset of 0 and 1s
distances = spd.pdist(data_01.iloc[:,1:], metric="euclidean")

#distances returns a condensed array, making it square was computationally ineffectient
# this calculates a list of tuples that map the condensed array to the standard Xij locations
tuples = (list(it.combinations(range(data_01.shape[0]),2)))


#Write out results to a CSV, to be graphed in R. This was pretty ineffecient
'''
with open('binary_dist.txt', 'w') as file_out:
    #iterate through list of tuples, each tuple containing a row and column index for data_01 array
    for i in range(0, len(tuples)):
        #use tuple to check whether row and column labels are the same (genuine) or differ (impostor)
        result = str((data_01.iat[tuples[i][0], 0] == data_01.iat[tuples[i][1], 0]))
        #write out comparison results along with the corresponding distance
        file_out.write(result + "," + str(distances[i])+ "\n")
'''

# make an array with just the labels
label_ar = np.array((data_01.iloc[:,0]))

# Take the array of labels (0 and 1) and calculate outer product with itself (using == as the operator).
# The resulting array is true for "genuine" matches and false for "impostor" matches
matches = (label_ar[:, np.newaxis] == label_ar).astype(int)

#calculate measurements for ROC cuve
genuine_rate, impostor_rate, thresholds = skm.roc_curve(matches.flatten(), spd.squareform(distances).flatten())

#calculator false_impostor_rate, equivalent to fnr
false_impostor_rate=1-genuine_rate

#plot the Genuine to Impostor rate, add EER line
mp.plot(genuine_rate, impostor_rate, lw=3)
mp.title("Impostor relative to Genuine over increasing Distance")
mp.plot([1, 0], [0,1], color='navy', lw=1, linestyle='--')
mp.xlabel('Genuine')
mp.ylabel('Impostor')
mp.show()

#calculate and display EER
EER = impostor_rate[np.nanargmin(np.absolute((false_impostor_rate - impostor_rate)))]
print("EER Calc: " + str(EER))
