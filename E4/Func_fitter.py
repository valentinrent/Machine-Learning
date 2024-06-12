import numpy as np
from SimpleNN import Func_Aprox_NN
import matplotlib.pyplot as plt


#create x vector with 1000 values between -1 and 1
x = np.linspace(-1,1,1000)

#create y labels using a polynomial function
y = 3*x**5 + 1.5*x**4 + 2*x**3 + 7*x + 0.5

#hyperparameters
lr = 0.01
epochs = 1000

#initialize model
func_model = Func_Aprox_NN()


#train model
def train_model(epochs, inputs, labels, model):
    losshistory = []
    for e in range(epochs):
        print(f'Epoch: {e}/{epochs}')

        for input,label in zip(inputs, labels):
            #forward pass
            output = model.forward(input)
            
            #calculate loss
            loss = model.MSE(label, output)
            
            #backwards pass
            model.backwards()
            
            #update weights
            model.optimize()
            
        losshistory.append(loss)
    
    return losshistory


loss = train_model(epochs, x, y, func_model)

predicted = func_model.predict(x)

#plot function x and y 
plt.figure(1)
plt.plot(x, y, color='blue', label='Actual')
plt.plot(x, predicted, color='red', label='Predicted')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network vs Actual Values')

plt.figure(2)
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Loss convergence over epochs for y = 3x^5 + 1.5x^4 + 2x^3 + 7x + 0.5')


plt.show()


