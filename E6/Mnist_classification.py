from SimpleNN import MNIST_NN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import random


#Get dataset
mnist = fetch_openml('mnist_784')

X, y = mnist['data'], mnist['target'].astype(np.int32)

X = X / 255.0

X = X.to_numpy()

y = np.eye(10)[y]

#split data and labels in train and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


#define input layer size
inputlayersize = X.shape[1]

#hyperparameters
lr = 0.01
epochs = 8

#initialize model
mnist_model = MNIST_NN(lr, inputlayersize, 64, 64, 10)


#train model
def train_model(epochs, inputs, labels, model):
    losshistory = []
    print(f'Start training for {epochs} epochs')
    for e in range(epochs):
        print(f'Epoch: {e}/{epochs}')

        for input,label in zip(inputs, labels):
            #forward pass
            output = model.forward(input)
            
            #calculate loss
            loss = model.cross_entropy_loss(label, output)
            
            #backwards pass
            model.backwards()
            
            #update weights
            model.optimize()
            
        losshistory.append(loss)
    
        print(f'CEL: {loss}')

    return losshistory

def test_accuracy(inputs, labels, model):
    print("Running test accuracy ...")
    total = 0
    right = 0 
    for input,label in zip(inputs, labels):
            
            #predict based on model
            output = model.predict(input)
            if np.argmax(output) == np.argmax(label):
                right += 1
            
            total += 1
    
    return right/total
    

def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss convergence over epochs')



def imshow_pred(inputs, labels, model):
    print("Displaying random predictions...")
    random_indices = random.sample(range(len(inputs)), 3)
    for index in random_indices:
        input = inputs[index]
        label = labels[index]
        
        # Predict based on model
        output = model.predict(input)
        predicted_label = np.argmax(output)
        label = np.argmax(label)
        
        # Reshape input to original image dimensions
        image = input.reshape(28, 28)
        
        # Display the image and predicted label
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"Actual Label: {label}\nPredicted Label: {predicted_label}")
        plt.axis('off')
        


#train model
loss = train_model(epochs, X_train, y_train, mnist_model)

#test model and get accuracy
acc = test_accuracy(X_test, y_test, mnist_model)

print(f'Accuracy: {acc*100}%')

#plot training loss
plot_loss(loss)

#show three random samples with prediction and label
imshow_pred(X_test, y_test, mnist_model)

plt.show()


