import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
y = training["Survived"]
features = ["Pclass", "Age", "Sex", "SibSp", "Parch", "IsChild", "Fare"]
X = pd.get_dummies(training[features])

#set classifier
classifier = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)
classifier_ = classifier.fit(X, y)

#Score model
print(classifier_.score(X,y))

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

X_test = pd.get_dummies(test[features])

results = classifier.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': results})

print(output['Survived'].value_counts(normalize=True))

output.to_csv('predictions.csv', index=False)