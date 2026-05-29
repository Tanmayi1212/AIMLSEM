import sys
sys.path.append('C:\\Users\\91703\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages') # Use the actual path
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score 
 
data = pd.read_csv('diabetes.csv') 
 
X = data.drop('Outcome', axis=1) 
y = data['Outcome'] 
 
X.fillna(X.mean(), inplace=True)
 
scaler = StandardScaler() 
X = scaler.fit_transform(X) 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
model = LogisticRegression() 
model.fit(X_train, y_train) 
 
y_pred = model.predict(X_test) 
 
accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred) 
recall = recall_score(y_test, y_pred) 
 
print(f'Accuracy: {accuracy:.2f}') 
print(f'Precision: {precision:.2f}') 
print(f'Recall: {recall:.2f}')