import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hw1_modules import *

data = pd.read_csv("train.csv", dtype={"label" : "category"})

#manually looked up one of each digit, these are the index numbers
digit_source_array = np.array([1,0,16,7,3,8,21,6,10,11])

#create data frame with initial index numbers
digit_test = pd.DataFrame(np.concatenate([np.array([0,1,2,3,4,5,6,7,8,9]).reshape((10,1)), digit_source_array.reshape((10,1)), np.full([10,1], np.NaN),np.full([10,1], np.inf),np.full([10,1], np.nan)], axis=1), columns=["Digit", "SourceIdx", "MatchIdx", "MatchDist", "MatchLabel"])
#cast digit column to int (from float)
digit_test["Digit"] = digit_test["Digit"].astype(int)

print(digit_test)
print(digit_test.iloc[3,1])

#iterate through the digits
for i in range (0, digit_test.shape[0]):
    print("Evaluating " + str(i))
    for j in range (0, data.shape[0]):
#        print(str(i) + " , " + str (j))
        distance = l2distance(data.iloc[int(digit_test.iat[i,1])], data.iloc[j])
        if distance == 0:
            continue
        elif distance < digit_test.iat[i,3]:
            digit_test.iat[i,3] = distance
            digit_test.iat[i,2] = j
            digit_test.iat[i,4] = data.iat[j,0]
        else:
            continue


print(digit_test)