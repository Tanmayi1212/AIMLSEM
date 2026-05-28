import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def gradient_descent(X,y,alpha=0.01,iterations=1000):
  m,n=X.shape
  theta=np.zeros(n)
  cost_history=[]

  for _ in range(iterations):
    predictions=X.dot(theta)
    errors=predictions-y

    gradient=(1/m)*X.T.dot(errors)

    theta-=gradient*alpha

    cost=(1/2*m)*(np.sum(errors**2))
    cost_history.append(cost)
  return theta,cost_history
data=pd.read_csv('diabetes.csv')
X=data.drop('Outcome',axis=1)
y=data['Outcome']

X.fillna(X.mean(),inplace=True)
scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
theta,cost_history=gradient_descent(X_train,y_train)
print("Theta values: ")
print(theta)