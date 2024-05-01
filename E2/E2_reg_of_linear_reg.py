# Description: This file is used to implement multivariate linear regression with regression using python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

#extraction of dataset
data_set = pd.read_csv("Real_estate.csv")
X = data_set.iloc[:, 1:-3].values
y = data_set.iloc[:, -1].values

#Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#splitting the data
split_ratio = 0.8
split_index = int(split_ratio * len(X))
x_train, x_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]



#initialize the hyper-parameters
lr = 0.001
reg_pram = 5
iterations = 10000
n = X.shape[1] + 1 #number of features + bias 
thetas = np.random.rand(n) 



#implement the gradient descent algorithm
def gradientDescent_with_reg(thetas, iterations, lr, X, y):
    losshistory = []
    paramhistory = np.empty((0,n))
    m = len(y)
    for i in range(iterations):
        predict = 0
        for i in range(n):
            if i == 0:
                predict = thetas[i]
            else:
                predict += thetas[i]*X[:,i-1]
        losshistory.append(1/(2*m)*(np.sum(np.square(predict-y)+reg_pram*np.sum(np.square(thetas)))))
        paramhistory = np.vstack([paramhistory, thetas])
        #update the parameters using stochastic gradient descent
        for i in range(n):
            if i == 0:
                thetas[i] = thetas[i] - lr * (1/m) * np.sum(predict-y)
            else:
                thetas[i] = thetas[i] - lr * (1/m) * (X[:,i-1].T.dot(predict-y) + reg_pram*np.square(thetas[i]))
    return thetas ,losshistory, paramhistory
    

thetas, losshistory, paramhistory = gradientDescent_with_reg(thetas, iterations, lr, x_train, y_train)



print(thetas)
print("MSE: ",losshistory[-1])


#plot the loss function
plt.figure(1)
x_plot = np.arange(iterations)

plt.plot(x_plot, losshistory, label='loss function', color='red')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss function over iterations')
plt.legend()


#show convergence of thetas
plt.figure(2)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
for i in range(n):
    plt.plot(x_plot, paramhistory[:,i], label='theta'+str(i), color=colors[i % len(colors)])
    plt.xlabel('Iterations')
    plt.ylabel('Theta')
    plt.title('Thetas over iterations')
    plt.legend()

#compare test and train data fitting
plt.figure(3)
y_train_predict = np.dot(x_train, thetas[1:]) + thetas[0]
y_test_predict = np.dot(x_test, thetas[1:]) + thetas[0]
x_index_test = np.arange(len(y_test))
x_index_train = np.arange(len(y_train))
plt.plot(x_index_train, y_train, label='train', color='blue')
plt.plot(x_index_train, y_train_predict, label='train predict', color='red')

plt.figure(4)
plt.plot(x_index_test, y_test, label='test', color='green')
plt.plot(x_index_test, y_test_predict, label='test predict', color='orange')


# Anzeigen der figures
plt.show()

