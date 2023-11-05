'''
Date: 07/05/2023

'''
##-----Import Data------
##Import pandas to process csv file as dataframe  
import numpy as np   
import pandas as pd  
df = pd.read_csv('task1.csv')
col_list = df.columns
##------- Process data - convert "Y" and "N" to 1 and 0 ---------
X = np.array(df[col_list[1:6]])
y = np.array(df[col_list[6]]).tolist()
#Number of Data
num = len(y)
for i in range(num):
    if y[i] == "Y":
        y[i] = 1
    else:
        y[i] = 0
    for j in range(5):
        if X[i,j] == "Y":
            X[i, j] = 1
        else:
            X[i, j] = 0           
#y = np.reshape(y, (num, 1))
## -----Build Decision Tree Model using Skleran and Create Training data and testing data -----
import sklearn
from sklearn.tree import DecisionTreeClassifier
#Import function to split data into training and validation
from sklearn.model_selection import train_test_split
#Import accuracy_score to estimate the accuracy of model
from sklearn.metrics import accuracy_score
## Get the number of features
num_features = X.shape[1]
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''Build the DecisionTree Model which has number of features as max_depth
   Because We should try to classify with each features 
'''
# Build Decision Tree Model
clf = DecisionTreeClassifier(max_depth = 2 * num_features)
#--------------Train and Estimate the Model ---------
#Training
clf.fit(X_train, y_train)
#Predict and estimate accuracy
y_pred = clf.predict(X_test).tolist()
accuracy = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        accuracy += 1
accuracy /= len(y_test) 

print("Accuracy:", accuracy)