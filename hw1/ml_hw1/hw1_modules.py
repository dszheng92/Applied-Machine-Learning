import numpy as np
import pandas as pd
import scipy.spatial.distance as spd
import scipy.stats as sps
import matplotlib.pyplot as plt

def displayDigit(row):
    '''Takes a row from our data set, and displays the original image it represents'''
    #Create an 1x784 array out of the greyscale values
    digit_array = np.array(row)
    #Resize array to be square
    digit_array.resize((28,28))
    #Cast values from int to floats, and plot using a greyscale map
    plt.imshow(digit_array.astype(np.float32), cmap="binary")
    plt.show()

#ended up unused since it was slower than scipy routines
def l2distance(item1, item2):
    '''each item is 785 element series, label first then data. Calculate the L2 distances between them'''
    return np.sqrt(sum(np.square(abs(np.array(item1[1:]) - np.array(item2[1:])))))

def knn_predict_class(trainvalues, trainlables, test, k):
    '''takes array of feature vectors for training, an array of observed values for the training feature vectors, 
    an array of testing vectors without labels, and an integer k for number of neighbors. Returns an array of predicted
    labels'''

    #initialize predicted labels vector
    predictions = np.ones(test.shape[0], dtype=int)
    print("Calculating distances")

    #calculate L2 distance between each training element and each testing element
    distances = spd.cdist(trainvalues, test)

    #iterate through each test instance
    for i in range(0,test.shape[0]):

        #Output to show how much progress is being made
        if i % 100 == 0:
            print("Calculating match " + str(i))

        #create array where first col is training label, second col is calculated distance between the current test element
        # and each training element

        matches = np.concatenate((trainlables, np.expand_dims((distances[:,i]), axis=1)), axis=1)

        #partition sort (argpartition) the match array on distance column, where row k is guaranteed to have all
        #smaller distances above and all larger distances below
        matches = matches[np.argpartition(matches[:, 1], k - 1, axis=0).transpose()]

        #find the most frequent label among the k closest neighbors, and use that as predicted label for test element
        predictions[i] = sps.mode(matches[:k-1,0])[0]

    #Return a vertical array of predicted labels for each test element
    return np.expand_dims(predictions.transpose(), axis=1)