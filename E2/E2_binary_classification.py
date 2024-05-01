# Description: This file is used to implement binary classification using logistic regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

#extraction of dataset
data_set = pd.read_csv("Real_estate.csv")
X = data_set.iloc[:, 1:-3].values
y_price = data_set.iloc[:, -1].values

#create y_lables where 1 means affordable or lower than the mean and 0 means unaffordable or higher than the mean
y_mean = np.mean(y_price)
y_lable = np.empty(len(y_price))
for i in range(len(y_price)):
    if y_price[i] > y_mean:
        y_lable[i] = 0
    else:
        y_lable[i] = 1


#Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#splitting the data
split_ratio = 0.8
split_index = int(split_ratio * len(X))
x_train, x_test = X[:split_index], X[split_index:]
y_train, y_test = y_lable[:split_index], y_lable[split_index:]



#initialize the hyper-parameters
lr = 0.01
iterations = 30000
n = X.shape[1] + 1 #number of features + bias 
thetas = np.random.rand(n) 



#implement the gradient descent algorithm
def gradientDescent_with_reg(thetas, iterations, lr, X, y):
    losshistory = []
    paramhistory = np.empty((0,n))
    m = len(y)
    for i in range(iterations):
        predict_lin = 0
        for i in range(n):
            if i == 0:
                predict_lin = thetas[i]
            else:
                predict_lin += thetas[i]*X[:,i-1]
        predict_sigmoid = 1/(1+np.exp(-predict_lin))
        losshistory.append(-1/(m)*(np.sum(y*np.log(predict_sigmoid)+(1-y)*np.log(1-predict_sigmoid))))
        paramhistory = np.vstack([paramhistory, thetas])
        #update the parameters using stochastic gradient descent
        for i in range(n):
            if i == 0:
                thetas[i] = thetas[i] - lr * (1/m) * np.sum(predict_sigmoid-y)
            else:
                thetas[i] = thetas[i] - lr * (1/m) * (X[:,i-1].T.dot(predict_sigmoid-y))
    return thetas ,losshistory, paramhistory
    
def pred_accuracy(thetas , X , y):
    examples = len(y)
    hit = 0
    for i in range(n):
            if i == 0:
                predict_lin = thetas[i]
            else:
                predict_lin += thetas[i]*X[:,i-1]
    predict_final = 1/(1+np.exp(-predict_lin))
    predict_final = np.round(predict_final) # round the prediction to 0 or 1
    for i in range(examples):
        if predict_final[i] == y[i]:
            hit += 1
        else:
            pass
    return (hit/examples)*100



thetas, losshistory, paramhistory = gradientDescent_with_reg(thetas, iterations, lr, x_train, y_train)



print(thetas)
print("MSE: ",losshistory[-1])
print("Training accuracy: ", pred_accuracy(thetas, x_train, y_train),"%")
print("Test accuracy: ", pred_accuracy(thetas, x_test, y_test),"%")


#plot the loss function
plt.figure(1)
x_plot = np.arange(iterations)

plt.plot(x_plot, losshistory, label='training loss function', color='red')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss function over iterations')
plt.legend()



plt.figure(2)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
for i in range(n):
    plt.plot(x_plot, paramhistory[:,i], label='theta'+str(i), color=colors[i % len(colors)])
    plt.xlabel('Iterations')
    plt.ylabel('Theta')
    plt.title('Thetas over iterations')
    plt.legend()

# Anzeigen der figures
plt.show()

