import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

def group_data(data):
    grouped_data = data.groupby(['Sex', 'Pclass', 'Title'])
    grouped_median = grouped_data.median()
    grouped_median = grouped_median.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
    
    return grouped_median

def find_age(row, data):
    condition = (
        (data['Sex'] == row['Sex']) &
        (data['Title'] == row['Title']) &
        (data['Pclass'] == row['Pclass']))

    if np.isnan(data[condition]['Age'].values[0]):
        condition = (
            (data['Sex'] == row['Sex']) &
            (data['Pclass'] == row['Pclass']))
    
    return data[condition]['Age'].values[0] 

def fix_age(data):
    data['Age'] = data.apply(lambda row:find_age(row, group_data(data)) if np.isnan(row['Age']) else row['Age'], axis=1)

    return data

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

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

    data = fix_age(data)

    data.Cabin.fillna('U', inplace = True)
    data['Cabin'] = data['Cabin'].map(lambda c: c[0])

    data.loc[data["Age"] < 13, "AgeGroup"] = 'Child'
    data.loc[(data["Age"] >= 13) & (data["Age"] <= 20), "AgeGroup"] = 'YoungAdult'
    data.loc[(data["Age"] > 20) & (data["Age"] < 65), "AgeGroup"] = 'Adult'
    data.loc[data["Age"] > 65, "AgeGroup"] = 'Senior'

    data["Family"] = data["SibSp"]+data["Parch"]
    data.loc[data['Family'] == 0, "FamilySize"] = 'Alone'
    data.loc[(data['Family'] > 0) & (data['Family'] < 5), "FamilySize"] = 'SmallFamily'
    data.loc[data['Family'] >= 5, "FamilySize"] = 'LargeFamily'

    return data

#Read in the training dataset
training  = pd.read_csv("train.csv")

#Clean and derive new data
training = clean_data(training)

#Set targets and features
y = training["Survived"]
features = ["Pclass", "Age", "Sex", "SibSp", "Parch", "FamilySize", "Title", "AgeGroup", "Fare", "Embarked", "Cabin"]
X = pd.get_dummies(training[features])

#set classifier
classifier = RandomForestClassifier(n_estimators=125, min_samples_leaf=3, max_depth=10, max_features=0.5)
classifier_ = classifier.fit(X, y)

importance = rf_feat_importance(classifier, X)
print(importance)
indexNames = importance[importance['imp'] >= 0.01].index
print(indexNames)
importance = importance.drop(indexNames)

X = X.drop(importance['cols'], axis=1)

classifier_ = classifier.fit(X, y)

#Score model
print(classifier_.score(X,y))

#Read in the training dataset
test  = pd.read_csv("test.csv")

ids = test["PassengerId"]
ids = ids.to_frame()

test = clean_data(test)

X_test = pd.get_dummies(test[features])

X_test["Cabin_T"] = 0

X_test = X_test.drop(importance['cols'], axis=1)

results = classifier.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': results})

print(output['Survived'].value_counts(normalize=True))

output.to_csv('predictions.csv', index=False)