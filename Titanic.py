import pandas as pd
import numpy as np
from sklearn import linear_model

#Read in the training dataset
training  = pd.read_csv("train.csv")

training["Fare"] = training["Fare"].fillna(training["Fare"].dropna().median())
training["Age"] = training["Age"].fillna(training["Age"].dropna().median())
training["Deck"] = training["Cabin"].str[:1]

training.loc[training["Sex"] == "male", "Sex"] = 0
training.loc[training["Sex"] == "female", "Sex"] = 1

training.loc[training["Age"] < 18, "IsChild"] = 1
training.loc[training["Age"] >= 18, "IsChild"] = 0

target = training["Survived"].values
features = training[["Pclass", "Age", "Sex", "SibSp", "Parch", "IsChild"]].values

classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)

print(classifier_.score(features,target))

print(training["Deck"].value_counts())
