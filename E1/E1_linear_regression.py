# Description: This file is used to implement linear regression using python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#create own dataset
dataSet = np.array([[1,4],[2,6.5],[1.5,5.2],[5,9.8],[3.2,7.4],[8.3,14.5],[6,11.5],[7.1,13.2],[8.2,15.4],[9.8,17.2],[11,22.3],[12,20.2],[13,23.2],[14,25.1],[15,27.2],[16,28.4],[17,31.5],[18,33.2],[19,35.4],[20,37.2],[21,39.8],[22,41.2],[23,43.2],[24,45.1],[25,47.2],[26,48.4],[27,51.5],[28,53.2],[29,55.4],[30,57.2],[31,59.8],[32,61.2],[33,63.2],[34,65.1],[35,67.2],[36,68.4],[37,71.5],[38,73.2],[39,76.4],[40,75.2],[41,78.8],[42,80.2],[43,83.2],[44,84.1],[45,90.2],[46,90.4],[47,93.5],[48,90.2],[49,97.4],[50,95.2]])
X = dataSet[:,0]
y = dataSet[:,1]
#split the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=5)

#initialize the hyper-parameters
lr = 0.0001
iterations = 500
theta_0 = 1
theta_1 = 1

#implement the gradient descent algorithm
def gradientDescent(theta_0, theta_1, iterations, lr, X, y):
    losshistory = []
    m = len(y)
    for i in range(iterations):
        predict = theta_0 + X * theta_1
        losshistory.append(1/(2*m)*np.sum(np.square(predict-y)))
        #update the parameters using stochastic gradient descent
        theta_0 = theta_0 - lr * (1/m) * np.sum(predict-y)
        theta_1 = theta_1 - lr * (1/m) * X.T.dot(predict-y)
    return theta_0, theta_1, losshistory
    

theta_0, theta_1, losshistory = gradientDescent(theta_0, theta_1, iterations, lr, x_train, y_train)

print("MSE: ",losshistory[-1])

# Scatter plot der tatsächlichen Werte
plt.scatter(x_train, y_train, label = 'actual values') #Here training values are used for the sake of more data points. The regression line also lines up with the test data.
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Scatter plot of actual values')

# Zeichnen der Geraden und der tatsächlichen Werte
plt.figure(1)
max_round_x = np.round(np.max(x_train))
x_values = np.arange(max_round_x+1)
y_values = theta_0 + theta_1 * x_values
plt.plot(x_values, y_values, label='regression line', color='red')
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

