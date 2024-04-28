#Notes from Lab1 - first task:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#extraction of dataset
data_set = pd.read_csv("Real_estate.csv")
feature = "X1 transaction date"
X, y = data_set[[feature]], data_set['Y house price of unit area']




#Normalize input data - splitting the train/test data is missing
scaler = StandardScaler()
X = scaler.fit_transform(X.values)
y = (y.to_numpy()).reshape(-1,1)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

#initialize the hyper-parameters
theta_0 = 1
theta_1 = 1
lr = 0.01
iteration = 10000
def gradientDescent(theta_0, theta_1, lr, X, y, iteration):
    m = len(y)
    loss_history = []
    for i in range(iteration):
        predict = theta_0 + X * theta_1
        loss_history.append((1/(2*m))*np.sum(np.square(predict-y)))
        theta_1 = theta_1 - lr * (1/m)*(X.T.dot(predict-y))
        theta_0 = theta_0 - lr * (1/m)*np.sum(predict-y)
    return theta_0, theta_1, loss_history

theta_0, theta_1, loss = gradientDescent(theta_0, theta_1, lr, X_train, Y_train, iteration)

Y_pred = X_test * theta_1 + theta_0

plt.scatter(X_test, Y_test, label = 'predcited values')
plt.plot(X_test, Y_pred, color='red')
plt.xlabel('Actual')
plt.ylabel('predicted')
plt.show()
