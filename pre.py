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
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split



df= pd.read_csv('benard file.csv')
# print(df.head())

df = df[["YEAR", "CE", "CI", "EI", "CS"]]
predict = "CE"

print(df.head())
# df.info()
# The describe() function summarizes the dataset’s statistical properties, such as count, mean, min, and max:

# print(df.describe())

# The corr() function displays the correlation between different variables in our dataset:
# df.corr()

# predict = "" #prediction value year.....etc
x = np.array(df.drop([predict], 1)) # Features
y = np.array(df[predict]) # Labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#ROUGH
# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print("Accuracy: " + str(acc))


# with open("CO2_Emission_Model.pickle", "wb") as f:  #save train modele with high acc
#     pickle.dump(linear, f)
pickle_in = open("CO2EmissionTrainModel.pickle", "rb")

linear = pickle.load(pickle_in)
print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")
predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print('CE', predicted[x], x_test[x], y_test[x])




####################

# # TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
# best = 0
# for _ in range(20):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#     linear = linear_model.LinearRegression()

#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print("Accuracy: " + str(acc))

#     if acc > best:
#         best = acc
#         with open("pred.pickle", "wb") as f:
#             pickle.dump(linear, f)

# # LOAD MODEL
# pickle_in = open("pred.pickle", "rb")
# linear = pickle.load(pickle_in)

# print("-------------------------")
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)
# print("-------------------------")

# predicted= linear.predict(x_test)
# for x in range(len(predicted)):
#     print(predicted[x], x_test[x], y_test[x])


# # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
# # linear = linear_model.LinearRegression()
# # linear.fit(x_train, y_train)
# # acc = linear.score(x_test, y_test) # acc stands for accuracy 
# # print('Coefficient: \n', linear.coef_) # These are each slope value
# # print('Intercept: \n', linear.intercept_) # This is the intercept
# # predictions = linear.predict(x_test) # Gets a list of all predictions


# x = pd.YEAR
# y = pd.CS
# plt.plot(x, y)
# plt.show()

