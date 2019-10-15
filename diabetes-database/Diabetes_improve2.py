# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

dataset = pd.read_csv('diabetes.csv')
dataset.head()
dataset.info()

#missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(dataset)

#Looking for correlation
corr_matrix = dataset.corr()
corr_matrix['Outcome'].sort_values(ascending=False)

#Devide Independent and dependent values.
x = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 8].values

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=0)

#StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(x,x['Glucose']):
    start_x_train = x.loc[train_index]
    start_x_test = x.loc[test_index]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
start_x_train = sc_x.fit_transform(start_x_train)
start_x_test = sc_x.transform(start_x_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
start_x_train = pca.fit_transform(start_x_train)
start_x_test = pca.transform(start_x_test)
explained_variance = pca.explained_variance_ratio_

#Creat Varies models

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(start_x_train, y_train)

#prediction for Logistic Regression
y_pred = classifier.predict(start_x_test)

#confution matrix for Logistic Regression
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate accuracy for Logistic Regression
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)*100)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(start_x_train, y_train)

#prediction for K-NN 
y_pred = classifier.predict(start_x_test)

#confution matrix for K-NN 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate accuracy for K-NN 
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)*100)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0,C=5,gamma=5)
classifier.fit(start_x_train, y_train)

#prediction for SVM
y_pred = classifier.predict(start_x_test)

#confution matrix for SVM 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate accuracy for SVM 
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)*100)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(start_x_train, y_train)

#prediction for Naive Bayes
y_pred = classifier.predict(start_x_test)

#confution matrix for Naive Bayes 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate accuracy for Naive Bayes 
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)*100)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(start_x_train, y_train)

#prediction for Decision Tree Classificatio
y_pred = classifier.predict(start_x_test)

#confution matrix for Decision Tree Classificatio 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate accuracy for Decision Tree Classificatio 
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)*100)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(start_x_train, y_train)

#prediction for Random Forest Classification
y_pred = classifier.predict(start_x_test)

#confution matrix for Random Forest Classification 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate accuracy for Random Forest Classification 
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)*100)




