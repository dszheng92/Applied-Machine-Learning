
# coding: utf-8

# In[80]:

get_ipython().magic('matplotlib inline')
import os
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import csv


# In[2]:

titanTrain = pd.read_csv("/Users/disheng/Desktop/Machine Learning/hw1/train.csv")
titanTest = pd.read_csv("/Users/disheng/Desktop/Machine Learning/hw1/test.csv")


# In[3]:

print(titanTrain.columns.values)


# In[4]:

#Replace the missing age with age median for a good estimate
titanTrain['Age'].fillna(titanTrain['Age'].median(), inplace=True)


# In[5]:

titanTrain.describe()


# In[6]:

titanTrain.info()
#Most of cabins are missing value. We will not use it as part of our analysis.


# In[7]:

survived = titanTrain[titanTrain['Survived']==1]['Sex'].value_counts()
dead = titanTrain[titanTrain['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived ,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
#Looks like gender plays a very important factor on survival rate. Women are much more likely to survice on Tiatanic. 


# In[8]:

ageRate = sns.FacetGrid(titanTrain, hue="Survived",aspect=4)
ageRate.map(sns.kdeplot,'Age',shade= True)
ageRate.set(xlim=(0, titanTrain['Age'].max()))
ageRate.add_legend()
#I looks like youngers (especially age less than 12) are likely to survive more than elderlies.


# In[9]:

Pclassrate = sns.FacetGrid(titanTrain, col='Survived', row='Pclass', size=2.2, aspect=1.6)
Pclassrate.map(plt.hist, 'Age', alpha=.5, bins=20)
Pclassrate.add_legend();
#Looks like Class 1 has the most survival rate, and Class 3 has the lowest.
#It can be seen that ticket fare has a stong correlation with passenger class. 


# In[10]:

survived1 = titanTrain[titanTrain['Survived']==1]['Pclass'].value_counts()
dead1 = titanTrain[titanTrain['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([survived1 ,dead1])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))


# In[11]:

#Use the most frequent embark type 'S' to fill NA values
titanTrain["Embarked"] = titanTrain["Embarked"].fillna("S")
#Create a proportion plot
sns.factorplot('Embarked','Survived', data=titanTrain,size=4,aspect=3)


# In[ ]:




# In[12]:

#Fare analysis. Fill NA fares value with median fare for better estimation 
titanTest["Fare"].fillna(titanTest["Fare"].median(), inplace=True)
titanTrain['Fare'] = titanTrain['Fare'].astype(int)
titanTest['Fare']    = titanTest['Fare'].astype(int)

#Estimate survival based one fare rate.
fare_dead = titanTrain["Fare"][titanTrain["Survived"] == 0]
fare_survive     = titanTrain["Fare"][titanTrain["Survived"] == 1]
fareAvg = DataFrame([fare_dead.mean(), fare_survive.mean()])
fareSTD = DataFrame([fare_dead.std(), fare_survive.std()])

#Plot fare rate vs. survival. 
fareAvg.index.names = fareSTD.index.names = ["Survived"]
fareAvg.plot(yerr = fareSTD,kind ='bar',legend=False)
#Based on the graph, people with higher ticket price are more likely to survive Titanic. 


# In[13]:

PclassRate = plt.subplot()
PclassRate.set_ylabel('Average fare')
titanTrain.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = PclassRate)
#It can be seen that ticket fare has a stong correlation with passenger class. 


# In[14]:

#Combining siblings and kids into a category called travel with Family then drop SibSp and Parch
titanTrain['Family'] = titanTrain["Parch"] + titanTrain["SibSp"]
titanTrain['Family'].loc[titanTrain['Family'] > 0] = 1
titanTrain['Family'].loc[titanTrain['Family'] == 0] = 0
titanTest['Family'] =  titanTest["Parch"] + titanTrain["SibSp"]
titanTest['Family'].loc[titanTest['Family'] > 0] = 1
titanTest['Family'].loc[titanTest['Family'] == 0] = 0


# In[15]:

fig, (pic1,pic2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
sns.countplot(x='Family', data=titanTrain, order=[1,0], ax = pic1)

#plot survival rate vs. travel with or without family.
withFamily = titanTrain[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=withFamily, order=[1,0], ax = pic2)

pic1.set_xticklabels(["With Family","Alone"], rotation=0)
#It can be inferred that those who travel with their families had a higher chanse of survival 


# In[16]:

#Fill the Na values with appropriate values
titanTest['Age'].fillna(titanTest['Age'].median(), inplace=True)
titanTrain["Embarked"] = titanTrain["Embarked"].fillna("S")
lb = LabelEncoder()
titanTrain['Embarked'] = lb.fit_transform(titanTrain['Embarked'])
titanTest['Embarked'] = lb.fit_transform(titanTest['Embarked'])
titanTrain['Sex'] = lb.fit_transform(titanTrain['Sex'])
titanTest['Sex'] = lb.fit_transform(titanTest['Sex'])


# In[17]:

#Trim away irrelevant information would help to reduce the data size. It's better for computation. 
titanTrain = titanTrain.drop(['PassengerId','Name','Ticket','Cabin','Family'], axis=1)
titanTest = titanTest.drop(['Name','Ticket','Cabin', 'Family'], axis=1)
X_train = titanTrain.drop("Survived", axis=1)
Y_train = titanTrain["Survived"]
X_test  = titanTest.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape



# In[18]:

#Logistic rate
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


# In[33]:

# Use training data cross validation to train the model and see estimates
clf = logreg.fit(X_train, Y_train)
scores = cross_val_score(clf, X_train, Y_train, cv=10)
print (scores)
print (np.mean(scores))


# In[31]:

#The prediction rate for LogisticRegression
Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[86]:

# Getting Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanTrain.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

coeff_df


# In[103]:

predictions = pd.DataFrame(Y_pred)
np.savetxt('/Users/disheng/Desktop/submission.csv', predictions, delimiter=',')
np.savetxt('/Users/disheng/Desktop/22.csv', passengers, delimiter=',')

