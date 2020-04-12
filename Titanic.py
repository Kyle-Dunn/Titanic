import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data.loc[data["Name"].str.contains("Mr."), "Title"] = "Mr"
    data.loc[data["Name"].str.contains("Mrs."), "Title"] = "Mrs"
    data.loc[data["Name"].str.contains("Miss."), "Title"] = "Ms"
    data.loc[data["Name"].str.contains("Master."), "Title"] = "Master"
    data.loc[data["Name"].str.contains("Don."), "Title"] = "Royal"
    data.loc[data["Name"].str.contains("Dona."), "Title"] = "Royal"
    data.loc[data["Name"].str.contains("Rev."), "Title"] = "Special"
    data.loc[data["Name"].str.contains("Major"), "Title"] = "Special"
    data.loc[data["Name"].str.contains("Sir."), "Title"] = "Royal"
    data.loc[data["Name"].str.contains("Lady."), "Title"] = "Royal"
    data.loc[data["Name"].str.contains("Mme."), "Title"] = "Mrs"
    data.loc[data["Name"].str.contains("Col."), "Title"] = "Special"
    data.loc[data["Name"].str.contains("Ms."), "Title"] = "Ms"
    data.loc[data["Name"].str.contains("Countess."), "Title"] = "Royal"
    data.loc[data["Name"].str.contains("Capt."), "Title"] = "Special"
    data.loc[data["Name"].str.contains("Dr."), "Title"] = "Special"
    data.loc[data["Name"].str.contains("Mlle."), "Title"] = "Ms"
    data.loc[data["Name"].str.contains("Jonkheer."), "Title"] = "Royal"
    data.loc[(data["Title"] == 'Dr') & (data["Sex"] == 'female'), "Title"] = "Mrs"

    data["Embarked"] = data["Embarked"].fillna("S")

    data.loc[(data["Title"] == 'Master') & (data["Age"].isnull()), "Age"] = data.loc[data['Title'] == 'Master', "Age"].dropna().median()
    data.loc[(data["Title"] == 'Mr') & (data["Age"].isnull()), "Age"] = data.loc[data['Title'] == 'Mr', "Age"].dropna().median()
    data.loc[(data["Title"] == 'Ms') & (data["Age"].isnull()), "Age"] = data.loc[data['Title'] == 'Ms', "Age"].dropna().median()
    data.loc[(data["Title"] == 'Mrs') & (data["Age"].isnull()), "Age"] = data.loc[data['Title'] == 'Mrs', "Age"].dropna().median()
    data.loc[(data["Title"] == 'Royal') & (data["Age"].isnull()), "Age"] = data.loc[data['Title'] == 'Royal', "Age"].dropna().median()
    data.loc[(data["Title"] == 'Special') & (data["Age"].isnull()), "Age"] = data.loc[data['Title'] == 'Special', "Age"].dropna().median()

    data.loc[data["Age"] < 13, "Child"] = 1
    data.loc[data["Age"] >= 13, "Child"] = 0

    data.loc[data["Age"] >= 13, "YoungAdult"] = 1
    data.loc[data["Age"] <= 20, "YoungAdult"] = 0

    data.loc[data["Age"] > 20, "Adult"] = 1
    data.loc[data["Age"] < 65, "Adult"] = 0

    data.loc[data["Age"] >= 65, "Senior"] = 1
    data.loc[data["Age"] < 65, "Senior"] = 0

    data["Family"] = data["SibSp"]+data["Parch"]
    data["IsAlone"] = data['Family'] == 0

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    return data

#Read in the training dataset
training  = pd.read_csv("train.csv")

#Clean and derive new data
training = clean_data(training)

#Set targets and features
y = training["Survived"]
features = ["Pclass", "Age", "Sex", "SibSp", "Parch", "Family", "IsAlone", "Title","Child", "YoungAdult", "Adult", "Senior", "Fare", "Embarked"]
X = pd.get_dummies(training[features])

#set classifier
classifier = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=1)
classifier_ = classifier.fit(X, y)

#Score model
print(classifier_.score(X,y))

#Read in the training dataset
test  = pd.read_csv("test.csv")

ids = test["PassengerId"]
ids = ids.to_frame()

test = clean_data(test)

X_test = pd.get_dummies(test[features])

results = classifier.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': results})

print(output['Survived'].value_counts(normalize=True))

output.to_csv('predictions.csv', index=False)