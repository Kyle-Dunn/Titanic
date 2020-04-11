import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read in the training dataset
training  = pd.read_csv("train.csv")

fig = plt.figure(figsize=(18,6))

plt.subplot2grid((2,3), (0,0))
training.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("% Survived")

plt.subplot2grid((2,3), (0,1))
plt.scatter(training.Survived, training.Age, alpha=0.1)
plt.title("Survival and Age")

plt.subplot2grid((2,3), (0,2))
training.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survival and Class")

plt.show()
