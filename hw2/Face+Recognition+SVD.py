
# coding: utf-8

# In[1]:

import numpy as np
import sklearn
from scipy import misc
from matplotlib import pylab as plt
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
from functools import reduce
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from numpy import linalg as LA

get_ipython().magic('matplotlib inline')


# In[2]:

train_labels, train_data = [], []


# In[3]:

for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)


# In[4]:

print (train_data.shape, train_labels.shape)
plt.imshow(train_data[50, :].reshape(50,50), cmap = cm.Greys_r)
plt.show()


# In[5]:

X = train_data


# In[6]:

test_labels, test_data = [], []


# In[7]:

for line in open('./faces/test.txt'):
    im = misc.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])
test_data, test_labels = np.array(test_data, dtype=float), np.array(test_labels, dtype=int)


# In[8]:

Xtest = test_data


# In[9]:

print (Xtest.shape, test_labels.shape)
plt.imshow(Xtest[50, :].reshape(50,50), cmap = cm.Greys_r)
plt.show()


# In[10]:

µ = X.mean(axis=0)   


# In[11]:

#c, average face
plt.imshow(µ.reshape(50,50), cmap = cm.Greys_r)


# In[12]:

#d
Xnew = np.subtract(X, µ) 


# In[13]:

Xnew.shape


# In[14]:

plt.imshow(Xnew[50, :].reshape(50,50), cmap = cm.Greys_r)


# In[15]:

Xtestnew = np.subtract(Xtest, µ) 
Xtestnew.shape


# In[16]:

plt.imshow(Xtestnew[50, :].reshape(50,50), cmap = cm.Greys_r)
#test after subtraction


# In[99]:

U, s, V = np.linalg.svd(Xnew, full_matrices=True)


# In[106]:

for i in range(1, 11):
    plt.figure()
    plt.imshow(V[i, :].reshape(50,50), cmap = cm.Greys_r)


# In[101]:

S = np.zeros((540, 2500), dtype=complex)
S[:540, :540] = np.diag(s)
np.allclose(Xnew, U.dot(S.dot(V)))


# In[80]:

np.allclose(Xnew, (U.dot(S)).dot(V))


# In[82]:

(U.dot(S)).dot(V)


# In[102]:

rankr = []

#frobenius norm
for r in range(1, 201):

    Xr = np.dot(U[:,: r ], np.dot(S[: r,: r ], V[: r,:]))
    U[:,: r ].shape, V[: r,:].shape, S[: r,: r ] .shape, 
    rankr.append(np.linalg.norm((X-Xr)))


# In[103]:

r = list(range(1, 201))


# In[105]:

plt.plot(r, rankr,  lw = 2)
plt.xlabel('r', fontsize=14, color='red')
plt.ylabel('approximation error', fontsize=14, color='red')
plt.title("Low-rank Approximation")


# In[83]:

Utest, stest, Vtest = np.linalg.svd(Xtestnew, full_matrices=True)


# In[125]:

#a function to generate r -dimensional feature matrix F
def featureMatrix(r):
    F = np.dot(Xnew, np.transpose(V[: r,:]))
    return F


# In[110]:

X[0].shape


# In[86]:

featureMatrix(4)[:4,:]


# In[126]:

#a function to generate r -dimensional feature matrix F
def featureTest(r):
    Ftest = np.dot(Xtestnew, np.transpose(V[: r,:]))
    return Ftest


# In[88]:

featureMatrix(10).shape


# In[89]:

len(train_labels)


# In[133]:

logreg = LogisticRegression(multi_class='ovr')
clf = logreg.fit(featureMatrix(10), train_labels)
scores = cross_val_score(clf, featureMatrix(10), train_labels, cv=10)
print (scores)
print (np.mean(scores))


# In[134]:

logreg.score(featureTest(10), test_labels)


# In[122]:

logreg.score(featureMatrix(100), train_labels)


# In[129]:

accuracy = []
for r in range (1,201):
    logreg = LogisticRegression(multi_class='ovr')
    clf = logreg.fit(featureMatrix(r), train_labels)
    #Y_pred = clf.predict(featureTest(r))
    #accuracy.append(logreg.score(featureMatrix(r), train_labels))
    accuracy.append(logreg.score(featureTest(r), test_labels))


# In[130]:

r = list(range(1, 201))
plt.plot(r, accuracy,  lw=2)
plt.xlabel('r', fontsize=14, color='red')
plt.ylabel('Accuracy', fontsize=14, color='red')
plt.title("Prediction accuracy")


# In[31]:

M = np.matrix('1 0 3; 3 7 2; 2 -2 8; 0 -1 1; 5 8 7')


# In[131]:

accuracy


# In[32]:

MTM = M.dot(np.transpose(M))


# In[33]:

MMT = np.transpose(M).dot(M)


# In[34]:

w1, v1 = LA.eig(MTM)


# In[35]:

w1, v1


# In[36]:

w2, v2 = LA.eig(MMT)


# In[37]:

w2, v2


# In[38]:

U1, s1, V1 = np.linalg.svd(M, full_matrices=True)


# In[39]:

U1, s1, V1


# In[40]:

s1

