import pandas as pd
import numpy as np
from sklearn import linear_model

#Read in the training dataset
training  = pd.read_csv("train.csv")

#Clean and derive new data
training["Fare"] = training["Fare"].fillna(training["Fare"].dropna().median())
training["Age"] = training["Age"].fillna(training["Age"].dropna().median())
training["Family"] = training["SibSp"]+training["Parch"]

training.loc[training["Sex"] == "male", "Sex"] = 0
training.loc[training["Sex"] == "female", "Sex"] = 1

training.loc[training["Age"] < 18, "IsChild"] = 1
training.loc[training["Age"] >= 18, "IsChild"] = 0

#Set targets and features
target = training["Survived"].values
features = training[["Pclass", "Age", "Sex", "SibSp", "Parch", "IsChild", "Fare"]].values

#set classifier
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)

#Score model
print(classifier_.score(features,target))

#Read in the training dataset
test  = pd.read_csv("test.csv")

ids = test["PassengerId"]
ids = ids.to_frame()

test["Fare"] = test["Fare"].fillna(test["Fare"].dropna().median())
test["Age"] = test["Age"].fillna(test["Age"].dropna().median())
test["Family"] = test["SibSp"]+test["Parch"]

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test.loc[test["Age"] < 18, "IsChild"] = 1
test.loc[test["Age"] >= 18, "IsChild"] = 0

test = test.drop(columns=["Name", "Ticket", "Cabin", "Embarked", "PassengerId", "Family"])

results = classifier.predict(test)

ids.insert(1, "Survived", results, True)

ids.to_csv('predictions.csv', index=False)