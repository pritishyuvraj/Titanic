#Solution by: Pritish Yuvraj
#This a question appearing on Kaggle Website
#Here we are predicting your chances of survival on the Titanic Ship based
#upon the class your travelling, Age, Sex, etc.

import pandas as pd 
import csv
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train = pd.read_csv("train.csv");
test = pd.read_csv("test.csv");

train["fare"] = 0
test["fare"] = 0
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
train["sex"] = 0
test["sex"] = 0
train["familySize"] = 0
test["familySize"] = 0
train["child"] = 0
test["child"] = 0

train["child"][train["Age"] < 18] = 1
train["child"][train["Age"] >= 18] = 0
test["child"][test["Age"] < 18] = 1
test["child"][test["Age"] >= 18] = 0
train["fare"][train["Fare"] >= 81] = 1
train["fare"][train["Fare"] < 81] = 0
test["fare"][test["Fare"] >= 81] = 1
test["fare"][test["Fare"] < 81] = 0
train["sex"][train["Sex"] == 'male'] = 1
train["sex"][train["Sex"] == 'female'] = 0
test["sex"][test["Sex"] == 'male'] = 1
test["sex"][test["Sex"] == 'female'] = 0
train["familySize"] = train["Parch"] + train["SibSp"]
test["familySize"] = test["Parch"] + test["SibSp"]

featuresforest = train[["Pclass", "sex", "child", "familySize", "fare"]].values
featurestest = test[["Pclass", "sex", "child", "familySize", "fare"]].values

target = train["Survived"].values

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(featuresforest, target)
print my_forest.score(featuresforest, target)
print my_forest.feature_importances_

pred_soln = my_forest.predict(featurestest)

print len(pred_soln)

#Printing to file
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_soln, PassengerId, columns = ["Survived"])
my_solution.to_csv("pritish_yuv.csv", index_label = ["PassengerId"])