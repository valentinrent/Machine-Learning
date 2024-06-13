import SimpleNN
import matplotlib.pyplot as plt
import numpy as np



#create input and labels for AND training
inputs_AND = np.array([[0,0],[0,1],[1,0],[1,1]])
labels_AND = np.array([0,0,0,1])

#create input and labels for XOR training
inputs_XOR = np.array([[0,0],[0,1],[1,0],[1,1]])
labels_XOR = np.array([0,1,1,0])

#hyperparameters
lr = 0.1
epochs = 10000



#initialize models
AND_model = SimpleNN.Logic_Gate_NN()
XOR_model = SimpleNN.Logic_Gate_NN()

#train model
def train_model(epochs, inputs, labels, model):
    losshistory = []
    for e in range(epochs):
        print(f'Epoch: {e}/{epochs}')

        for input,label in zip(inputs, labels):
            #forward pass
            output = model.forward(input)
            
            #calculate loss
            loss = model.binary_log_loss(label, output)
            
            #backwards pass
            model.backwards()
            
            #update weights
            model.optimize()
            
        losshistory.append(loss)
    
    return losshistory



#Predict function

def predict(model, inputs, labels):
    for input,label in zip(inputs,labels):
        prediction = model.predict(input)
        print(f'Input: {input}, Prediction: {prediction}, Label: {label}')


#plot function
def plot_loss(name,losshistory):
    plt.figure()
    plt.plot(losshistory)
    plt.title(f'{name} Loss convergence over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    


loss_AND = train_model(epochs, inputs_AND, labels_AND, AND_model)
predict(AND_model, inputs_AND, labels_AND)
plot_loss("AND",loss_AND)

loss_XOR = train_model(epochs, inputs_XOR, labels_XOR, XOR_model)
predict(XOR_model, inputs_XOR, labels_XOR)
plot_loss("XOR",loss_XOR)


