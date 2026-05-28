import pandas as pd
from sklearn.linear_model import Lasso

data=pd.read_csv('Social_Network_Ads.csv')

X=data.drop(['Gender','Purchased'],axis=1)
y=data['Purchased']

def lasso_reg(X,y,alpha=1.0):
  lasso=Lasso(alpha=alpha)
  lasso.fit(X,y)
  return lasso.coef_
print(lasso_reg(X,y))