import pickle

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import sklearn
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
for i in range(2):
    print()

predict = "G3"

X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

# create train and test split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.1
)

# train model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(f"Accuracy: {acc}")

# save model
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# predict and display for each x
predictions = linear.predict(x_test)
print()
for x in range(len(predictions)):
    print(f"{predictions[x]}  {y_test[x]} {x_test[x]}")


p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()
