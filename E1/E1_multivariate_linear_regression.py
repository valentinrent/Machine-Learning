# Description: This file is used to implement multivariate linear regression using python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#create own dataset on base function: 𝑌 = 𝑎*𝑋 + 𝑏 + 𝑛𝑜𝑖𝑠𝑒 
vec_size = 200  
x_1 = np.random.uniform(0, 20, vec_size)
x_2 = np.random.uniform(0, 20, vec_size)
x_3 = np.random.uniform(0, 20, vec_size)
y = 3 + 0.5*x_1 + 1.5*x_2 + 2*x_3 + np.random.normal(0, 5, vec_size) #create y value and add noise
X = np.column_stack((x_1, x_2, x_3))

#split the dataset into training and testing data

x_train, x_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=5)


#initialize the hyper-parameters
lr = 0.001
iterations = 10000
theta_0 = 1
theta_1 = 1
theta_2 = 1
theta_3 = 1



#implement the gradient descent algorithm
def gradientDescent(theta_0, theta_1, theta_2, theta_3, iterations, lr, X, y):
    losshistory = []
    m = len(y)
    x_train_1 , x_train_2, x_train_3 = X[:,0], X[:,1], X[:,2]
    for i in range(iterations):
        predict = theta_0 + x_train_1 * theta_1 + x_train_2 * theta_2 + x_train_3 * theta_3
        losshistory.append(1/(2*m)*np.sum(np.square(predict-y)))
        #update the parameters using stochastic gradient descent
        theta_0 = theta_0 - lr * (1/m) * np.sum(predict-y)
        theta_1 = theta_1 - lr * (1/m) * x_train_1.T.dot(predict-y)
        theta_2 = theta_2 - lr * (1/m) * x_train_2.T.dot(predict-y)
        theta_3 = theta_3 - lr * (1/m) * x_train_3.T.dot(predict-y)
    return theta_0, theta_1, theta_2, theta_3 ,losshistory
    

theta_0, theta_1, theta_2, theta_3, losshistory = gradientDescent(theta_0, theta_1, theta_2, theta_3, iterations, lr, x_train, y_train)
print(theta_0, theta_1, theta_2, theta_3)
print("MSE: ",losshistory[-1])

#Show y-function over x and y_hat function over x
plt.figure(1)
x_plot = np.arange(len(x_test))
print(len(x_test))
y_plot = 0.5*x_test[:,0] + 1.5*x_test[:,1] + 2*x_test[:,2] + 3
y_hat = theta_0 + theta_1*x_test[:,0] + theta_2*x_test[:,1] + theta_3*x_test[:,2]
plt.plot(x_plot, y_plot, label='y', color='blue')
plt.plot(x_plot, y_hat, label='y_hat', color='red')

plt.legend() 

# Lossfunktion
plt.figure(2)
x_loss = np.arange(iterations)
plt.plot(x_loss, losshistory, label='loss function', color='red')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss function over iterations')
plt.legend()

# Anzeigen der figures
plt.show()

