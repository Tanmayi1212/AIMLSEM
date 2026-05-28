import pandas as pd
from sklearn.linear_model import Ridge

data=pd.read_csv('Social_Network_Ads.csv')

X=data.drop(['Gender','Purchased'],axis=1)
y=data['Purchased']

def ridge_reg(X,y,alpha=1.0):
  ridge=Ridge(alpha=alpha)
  ridge.fit(X,y)
  return ridge.coef_
print(ridge_reg(X,y))