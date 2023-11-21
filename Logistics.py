import pandas as pd


data = pd.read_csv('https://confrecordings.ams3.digitaloceanspaces.com/iris.data',names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])

#check all operations on left side
print(data.head())
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.isnull().sum())
print(data['Species'].nunique())
#Do the conversion
data['Species'].replace({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2 },inplace= True)

#Select Feature X and Y and print their values
X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']].values
Y = data['Species'].values
print(X)
print(Y)
#import libraries
from sklearn.model_selection import train_test_split

#divide data into training and testing and print their shape
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LogisticRegression

#load the model
softmax_reg = LogisticRegression(multi_class='multinomial')
softmax_reg.fit(X_train,Y_train)


Y_pred = softmax_reg.predict(X_test)
print(Y_pred)

from sklearn.metrics import accuracy_score

#print accuracy score
print((Y_test,Y_pred))

print(softmax_reg.predict_proba([[3.4,2.7,5.1,1.3]]))

#predict for single value
print(softmax_reg.predict_proba([[3.4,2.7,5.1,1.3]]))
import matplotlib.pyplot as plt
plt.plot(X[:, 0][Y==1], X[:, 1][Y==1], "y.", label="Iris-virginica")
plt.plot(X[:, 0][Y==0], X[:, 1][Y==0], "b.", label="Iris-Setosa")
plt.legend(loc="upper left", fontsize=14)
plt.show()
