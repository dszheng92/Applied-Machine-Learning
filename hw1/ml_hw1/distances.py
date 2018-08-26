import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
import hw1_modules

data = pd.read_csv("train.csv", dtype={"label" : "category"})


# index of digits 0-9 in data manually selected [1,0,16,7,3,8,21,6,10,11]

digits = data.iloc[[1,0,16,7,3,8,21,6,10,11],:]

#calculate euclidean/l2 distance between our array of test digits, and the rest of the training data
results = spd.cdist(digits, data, "euclidean")

#since our selected digits were not removed from data, they will match with their original entry with distance 0
#this changes those entries to infinite distance so they will not interfere with finding closest match
results[results==0] = np.inf

#Find the index of the minimum distance for each digit
mins_location = np.argmin(results, axis=1)

#create a DataFrame with a row for each test element, and columns showing the element label (which digit it is),
# the index of the element in our data, the index of the closest match in our data, and the label of the closest match
digit_test = pd.DataFrame(np.concatenate([np.array([0,1,2,3,4,5,6,7,8,9]).reshape((10,1)), np.array([1,0,16,7,3,8,21,6,10,11]).reshape((10,1)), np.full([10,1], np.NaN),np.full([10,1], np.nan)], axis=1), columns=["Digit", "SourceIdx", "MatchIdx", "MatchLabel"], dtype=int)

#Iterate through each digit
for i in range (0, 10):
    #Find the distance to closest match and store
    digit_test.iloc[i,2] = mins_location[i]
    #find the label of closest match and store
    digit_test.iloc[i,3] = data.iloc[mins_location[i],0]

#Display results
print(digit_test)
