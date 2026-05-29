import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confustion_matrix,classification_report
from sklearn.model_selection import train_test_split

digits=datasets.load_digits()

X=digits.images.reshape(len(digits),-1)
y=digits.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svc=SVC(kernel='rbf',gamma=0.001,c=100)

svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confustion_matrix(y_test,y_pred))

plt.figure(figsize=(10,4))

for index,(image,prediction) in enumerate(zip(X_test[:5],y_pred[:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(image.reshape(8,8),cmap=plt.cm.gray)
    plt.set_title(f"{prediction}")
    pt.axis('off')
plt.show()