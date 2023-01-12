import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle



df= pd.read_csv('benard file.csv')
# print(df.head())

df = df[["YEAR", "CE", "CI", "EI", "CS"]]
predict = "CE"

# print(df.head())
# df.info()
# The describe() function summarizes the dataset’s statistical properties, such as count, mean, min, and max:

# print(df.describe())

# The corr() function displays the correlation between different variables in our dataset:
# df.corr()

# predict = "" #prediction value year.....etc
x = np.array(df.drop([predict], 1)) # Features
y = np.array(df[predict]) # Labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#ROUGH
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))
    if acc > best:
        best = acc
        with open("CO2EmissionTrainModel.pickle", "wb") as f:  #save train modele with high acc
            pickle.dump(linear, f)

