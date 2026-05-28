import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def least_squares(X,y):
  return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
  
data=pd.read_csv('diabetes.csv')
X=data.drop('Outcome',axis=1)
y=data['Outcome']

X.fillna(X.mean(),inplace=True)
scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
return least_squares(X_train,y_train)