import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandradScaler

data=pd.read_csv('diabetes.csv')
X=data.drop('Outcome',axis=1)
y=data['Outcome']

X.fillna(X.mean(),inplace=True)
scaler=StandardScaler()
X=scaler.fit_transform(X)

def polynomial_regression(X,y,degree=3):
  poly=PolynomialFeatures(degree)
  X_poly=poly.fit_transform(X)
  theta=np.linalg.inv(X_poly.T.dot(X_ploy)).dot(X_poly.T).dot(y)
  return theta
print(polynomial_regression(X,y))