# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Elamaran S E
RegisterNumber:  212222230036
*/

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### data.head():
![275325718-0132c4fa-35e0-4b03-8383-52a06a5adba9](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/5a04ed0c-466c-44b5-8ca0-7825df32907b)
### data.info():
![275325749-79c50faa-26bf-4647-9540-2dd54615a67b](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/2ad31891-4f14-4b22-b0c2-b5df0c12b19d)
### isnull() and sum():
![275325786-cc1a8eea-2adf-4787-b698-341ac2a2e3a1](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/51bc4ba2-7e97-47b7-a76e-f2f430522bef)
### data value count():
![275325835-33705919-842d-4475-b31e-c2a75f9a09c0](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/aabd27b4-51bb-43ed-8b15-e805e5f8037f)
### data.head() for salary:
![275325908-b4277edd-d240-48ca-b180-7a215c882e48](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/0a24250d-f37e-42f4-ae8e-4c58427473a1)
### x.head():
![275325941-6ccdb4f3-4ccd-45c3-a9d5-1c2486b2f030](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/9e8d3197-2ea7-4f15-aa7c-b534e836c233)
### accuracy value:
![275325979-7aca4148-7acf-4d39-a9d7-50164dff6bb8](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/7a096b78-c705-464b-9c85-55d70fca47a5)
### data prediction:
![275326025-0b5c5cd2-9127-4b08-aaa4-94e0e6a42197](https://github.com/elamarannn/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497531/0bb99e88-c806-46f6-b988-b4bd229c2c69)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
