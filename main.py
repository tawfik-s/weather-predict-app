import math

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot, pyplot as plt
from seaborn import lineplot, displot
from sklearn.svm import SVC

df = pd.read_csv("seattle-weather.csv")
df.info()
sum1 = 0
sum2 = 0

count = 0
for x in df['precipitation']:
    if not math.isnan(x):
        sum1 += x
        count += 1

average = sum1 / count

df['precipitation'] = df['precipitation'].replace({np.nan: average})

df.info()

count = 0
for x in df['temp_max']:
    if not math.isnan(x):
        sum1 += x
        count += 1

average = sum1 / count

df['temp_max'] = df['temp_max'].replace({np.nan: average})

df.info()

# drop date from df

df = df.drop("date", axis=1)

## data exploring

import warnings

# print(df.shape)
# warnings.filterwarnings('ignore')
# sns.countplot("weather", data=df, palette="hls")

## data modeling
# lineplot(data=df)
# pyplot.show()

sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data=df, x="weather", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=df, x="temp_max", kde=True, ax=axs[0, 1], color='red')
sns.histplot(data=df, x="temp_min", kde=True, ax=axs[1, 0], color='skyblue')
sns.histplot(data=df, x="wind", kde=True, ax=axs[1, 1], color='orange')

plt.show()

data = df
countrain = len(data[data.weather == "rain"])
countsun = len(data[data.weather == "sun"])
countdrizzle = len(data[data.weather == "drizzle"])
countsnow = len(data[data.weather == "snow"])
countfog = len(data[data.weather == "fog"])
print("Percent of Rain:{:2f}%".format((countrain / (len(data.weather)) * 100)))
print("Percent of Sun:{:2f}%".format((countsun / (len(data.weather)) * 100)))
print("Percent of Drizzle:{:2f}%".format((countdrizzle / (len(data.weather)) * 100)))
print("Percent of Snow:{:2f}%".format((countsnow / (len(data.weather)) * 100)))
print("Percent of Fog:{:2f}%".format((countfog / (len(data.weather)) * 100)))

## run models
# to encode weather data
# lc = LabelEncoder()
# df["weather"] = lc.fit_transform(df["weather"])
print("weather unique values", df.weather.unique())

############################################

# x = ((df.loc[:, df.columns != "weather"]).astype(int)).values[:, 0:]
x = df.drop("weather", axis=1)
print(x)
y = df["weather"].values
print("yyyyyyyyyyyyyyyyyyyyyyyyyy")
print(y)

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=2)
print("*********************************************")
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print("KNN Accuracy:{:.2f}%".format(knn.score(x_test, y_test) * 100))
knnScore = knn.score(x_test, y_test) * 100

print("**********************************************")

svm = SVC()
svm.fit(x_train, y_train)
print("SVM Accuracy:{:.2f}%".format(svm.score(x_test, y_test) * 100))
svmAccuracy = svm.score(x_test, y_test) * 100
test = [[1.140175, 8.9, 2.8, 2.469818]]

print(svm.predict(test))

print("**********************************************")

# from sklearn import linear_model
#
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)
# print("linear regression Accuracy:{:.2f}%".format(reg.score(x_test, y_test) * 100))


# asociation rules naivebays

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)
# making predictions on the testing set
y_pred = gnb.predict(x_test)
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

NaiveBayesGaussianNBAccuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

print("**********************************************")

from sklearn import linear_model

logr = linear_model.LogisticRegression()
logr.fit(x_train, y_train)
y_pred = logr.predict(x_test)
LinearModelAccuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("logistic regression model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier

print("**********************************************")

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(x_test)
DecisionTreeClassifierAccuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("decision Tree Classifier model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

from tkinter import *


def calc(*args):
    # Get the values from the StringVar objects
    precipitation = float(v1.get())
    temp_max = float(v2.get())
    temp_min = float(v3.get())
    wind = float(v4.get())
    res = clf.predict([[precipitation, temp_max, temp_min, wind]])
    lmres = logr.predict([[precipitation, temp_max, temp_min, wind]])
    gnbres = gnb.predict([[precipitation, temp_max, temp_min, wind]])
    svmres = svm.predict([[precipitation, temp_max, temp_min, wind]])
    knnres = knn.predict([[precipitation, temp_max, temp_min, wind]])
    # Only change the text of the existing Label
    label2["text"] = res[0]
    label3["text"] = lmres[0]
    label4["text"] = gnbres[0]
    label5["text"] = svmres[0]
    label6["text"] = knnres[0]


master = Tk(className='weather predict app')
master.geometry("350x250")

# make this Label once
label2 = Label(master)
label2.grid(row=4, column=1)

label3 = Label(master)
label3.grid(row=5, column=1)

label4 = Label(master)
label4.grid(row=6, column=1)

label5 = Label(master)
label5.grid(row=7, column=1)

label6 = Label(master)
label6.grid(row=8, column=1)

Label(master, text="precipitation").grid(row=0, sticky=E)
Label(master, text="temp_max").grid(row=1, sticky=E)
Label(master, text="temp_min").grid(row=2, sticky=E)
Label(master, text="wind").grid(row=3, sticky=E)
Label(master, text=str("DecisionTreeClassifier " + str(round(DecisionTreeClassifierAccuracy, 2)))).grid(row=4, sticky=E)
Label(master, text=str("LinearModel " + str(round(LinearModelAccuracy, 2)))).grid(row=5, sticky=E)
Label(master, text=str("NaiveBayesGaussianNB " + str(round(NaiveBayesGaussianNBAccuracy, 2)))).grid(row=6, sticky=E)
Label(master, text=str("SupportVectorMachine " + str(round(svmAccuracy, 2)))).grid(row=7, sticky=E)
Label(master, text=str("k nearest neighbor " + str(round(knnScore, 2)))).grid(row=8, sticky=E)


# Create StringVars
v1 = StringVar()
v2 = StringVar()
v3 = StringVar()
v4 = StringVar()

e1 = Entry(master, textvariable=v1)
e2 = Entry(master, textvariable=v2)
e3 = Entry(master, textvariable=v3)
e4 = Entry(master, textvariable=v4)

# Trace when the StringVars are written
v1.trace_add("write", calc)
v2.trace_add("write", calc)
v3.trace_add("write", calc)
v4.trace_add("write", calc)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)

master.mainloop()
