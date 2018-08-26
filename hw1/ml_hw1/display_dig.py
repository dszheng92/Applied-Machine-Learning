from hw1_modules import *

data = pd.read_csv("train.csv", dtype={"label" : "category"})

#Looking for digits 0-9
for i in (range(0, 10)):
    #for each digit, look through rows of data
    for j in range(0,len(data)-1):
        #If the label in our current row matches the label we're looking for
        if int(data.iloc[j,:].loc["label"]) == i:
            #then display this digit using displayDigit function (in hw1_modules)
            displayDigit(data.iloc[j, 1:])
            break