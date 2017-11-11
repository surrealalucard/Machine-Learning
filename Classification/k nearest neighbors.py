import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

# Reading data and replacing empty (?) with -99999
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)

# Dropping id because its useless for our alg.
df.drop(['id'], 1, inplace=True)

# Setting up features (X) and labels (y)
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,1,4,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

# We are predicting whether it will be benign or malignant
prediction = clf.predict(example_measures)
print(prediction)
