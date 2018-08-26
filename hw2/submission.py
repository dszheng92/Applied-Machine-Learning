import json
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
with open('train.json') as json_data:
    d = json.load(json_data)

ingredients = []
cuisines = []

for i in range(0,len(d)):
    cuisines.append(d[i]["cuisine"].lower())
    for ingredient in d[i]["ingredients"]:
        ingredients.append(ingredient.lower())

cuisines = list(set(cuisines))
ingredients = list(set(ingredients))
print(len(d))
print(len(ingredients))
print(len(cuisines))
print(cuisines)

ingredient_dict = {ingredients[i] : i for i in range(0,len(ingredients))}
cuisine_dict = {cuisines[i] : i for i in range(0,len(cuisines))}
print(ingredient_dict)
print(cuisine_dict)

train_data = np.array(np.zeros([len(d), (2 + len(ingredients))]))
print(train_data.shape)
train_labels = pd.DataFrame(np.array(np.zeros([len(d),1])))

for i in range(0,len(d)):
    train_data[i][0] = d[i]["id"]
    train_labels.iloc[i] = d[i]["cuisine"].lower()
    for ingredient in d[i]["ingredients"]:
        train_data[i][2 + ingredient_dict[ingredient.lower()]] = 1


with open('test.json') as json_data:
    t = json.load(json_data)

test_data = np.array(np.zeros([len(t), (1 + len(ingredients))]))
for i in range(0,len(t)):
    test_data[i][0] = t[i]["id"]
    for ingredient in d[i]["ingredients"]:
        test_data[i][1 + ingredient_dict[ingredient.lower()]] = 1


lgr = LogisticRegression()
lgr_scores = cross_val_score(lgr, train_data[:,2:], train_labels[:].as_matrix().reshape(len(train_data)), cv=3)
print(lgr_scores)
lgr.fit(train_data[:,2:], train_labels[:].as_matrix().reshape(len(train_data)))
pred_labels = pd.DataFrame(lgr.predict(test_data[:,1:]))
print(pred_labels.shape)
print(pred_labels)
ids = pd.DataFrame(test_data[:,0], dtype=int)
print(ids.shape)
results = pd.concat([ids,pred_labels], axis=1)
print(results.shape)
print(results)
results.to_csv("results.csv", index=False, header=("id", "cuisine"))