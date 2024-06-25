import numpy as np






class Logic_Gate_NN:
    def __init__(self, lr = 0.1 ,input_layer_size=2, hidden_layer_size=2, output_layer_size=1):
        #initialize learning rate
        self.lr = lr

        #define layer sizes
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        #first layer weights and biases
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.b1 = np.random.randn(self.hidden_layer_size)

        #second layer weight and biases
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        self.b2 = np.random.randn(self.output_layer_size)

        #initialize activation buffers
        self.a_hidden = np.zeros(self.hidden_layer_size)
        self.a_output = np.zeros(self.output_layer_size)

        #initialize loss buffers
        self.l_hidden = np.zeros(self.hidden_layer_size)
        self.l_output = np.zeros(self.output_layer_size)

        #temporary input/output buffer
        self.input = None
        self.label = None


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    


    def binary_log_loss(self, label, prediction):
        self.label = label
        return -(label*np.log(prediction) + (1-label)*np.log(1-prediction))




    def forward(self, input):
        self.input = input
        #first layer
        self.a_hidden = self.sigmoid(np.dot(input, self.W1) + self.b1)
        
        #second layer
        self.a_output = self.sigmoid(np.dot(self.a_hidden, self.W2) + self.b2)

        return self.a_output


    def backwards(self):
        
        self.l_output = self.a_output - self.label

        self.l_hidden = np.dot(self.W2, self.l_output) * self.a_hidden * (1-self.a_hidden)


    def optimize(self):
        self.W2 -= self.lr * np.outer(self.a_hidden, self.l_output)
        self.b2 -= self.lr * self.l_output

        self.W1 -= self.lr * np.outer(self.input, self.l_hidden)
        self.b1 -= self.lr * self.l_hidden


    def predict(self, input):
        unrounded = self.forward(input)
        return np.round(unrounded).astype(int)

        


class Func_Aprox_NN:
    def __init__(self, lr = 0.00001 ,input_layer_size=1, hidden_layer_1_size=8,hidden_layer_2_size=8, output_layer_size=1):
        #initialize learning rate
        self.lr = lr

        #define layer sizes
        self.input_layer_size = input_layer_size
        self.hidden_layer_1_size = hidden_layer_1_size
        self.hidden_layer_2_size = hidden_layer_2_size
        self.output_layer_size = output_layer_size

        #first layer weights and biases
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_1_size)
        self.b1 = np.zeros((1,self.hidden_layer_1_size))
        

        #second layer weight and biases
        self.W2 = np.random.randn(self.hidden_layer_1_size, self.hidden_layer_2_size)
        self.b2 = np.zeros((1,self.hidden_layer_2_size))

        #third layer weight and biases
        self.W3 = np.random.randn(self.hidden_layer_2_size, self.output_layer_size)
        self.b3 = np.zeros((1,self.output_layer_size))

        #initialize activation buffers
        self.a_hidden_1 = np.zeros(self.hidden_layer_1_size)
        self.a_hidden_2 = np.zeros(self.hidden_layer_2_size)
        self.a_output = np.zeros(self.output_layer_size)

        #initialize loss buffers
        self.l_hidden_1 = np.zeros(self.hidden_layer_1_size)
        self.l_hidden_2 = np.zeros(self.hidden_layer_2_size)
        self.l_output = np.zeros(self.output_layer_size)

        #temporary input/output buffer
        self.input = None
        self.label = None


    def RelU(self,x):
        return np.maximum(0,x)
    
    def tanh(self, x):
        return np.tanh(x)

    def MSE(self, label, prediction):
        self.label = label
        return np.mean((label-prediction)**2)


    def forward(self, input):
        self.input = input

        #first layer
        self.a_hidden_1 = self.RelU(np.dot(input, self.W1) + self.b1)

        #second layer
        self.a_hidden_2 = self.RelU(np.dot(self.a_hidden_1, self.W2) + self.b2)
        
        #output layer
        self.a_output = np.dot(self.a_hidden_2, self.W3) + self.b3

        self.a_output = self.a_output.item()        

        return self.a_output


    def backwards(self):
        
        self.l_output = self.a_output - self.label

        self.l_hidden_2 = np.dot(self.W3, self.l_output) * (self.a_hidden_2>0).reshape(-1,1)
        
        self.l_hidden_1 = np.dot(self.W2, self.l_hidden_2) * (self.a_hidden_1>0).reshape(-1,1)
        

    def optimize(self):
        self.W3 -= self.lr * np.outer(self.a_hidden_2, self.l_output)
        self.b3 -= self.lr * self.l_output

        self.W2 -= self.lr * np.outer(self.a_hidden_1, self.l_hidden_2)
        self.b2 -= self.lr * self.l_hidden_2.reshape(1,-1)

        self.W1 -= self.lr * np.outer(self.input, self.l_hidden_1)
        self.b1 -= self.lr * self.l_hidden_1.reshape(1,-1)


    def predict(self, input):
        prediction = []
        for i in input:
            unrounded = self.forward(i)
            prediction.append(unrounded)
        return prediction







class MNIST_NN:
    def __init__(self, lr = 0.001 ,input_layer_size=784, hidden_layer_1_size=80,hidden_layer_2_size=64, output_layer_size=10):
        #initialize learning rate
        self.lr = lr

        #define layer sizes
        self.input_layer_size = input_layer_size
        self.hidden_layer_1_size = hidden_layer_1_size
        self.hidden_layer_2_size = hidden_layer_2_size
        self.output_layer_size = output_layer_size

        #first layer weights and biases
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_1_size)
        self.b1 = np.zeros((1,self.hidden_layer_1_size))
        

        #second layer weight and biases
        self.W2 = np.random.randn(self.hidden_layer_1_size, self.hidden_layer_2_size)
        self.b2 = np.zeros((1,self.hidden_layer_2_size))

        #third layer weight and biases
        self.W3 = np.random.randn(self.hidden_layer_2_size, self.output_layer_size)
        self.b3 = np.zeros((1,self.output_layer_size))

        #initialize activation buffers
        self.a_hidden_1 = np.zeros(self.hidden_layer_1_size)
        self.a_hidden_2 = np.zeros(self.hidden_layer_2_size)
        self.a_output = np.zeros(self.output_layer_size)

        #initialize loss buffers
        self.l_hidden_1 = np.zeros(self.hidden_layer_1_size)
        self.l_hidden_2 = np.zeros(self.hidden_layer_2_size)
        self.l_output = np.zeros(self.output_layer_size)

        #temporary input/output buffer
        self.input = None
        self.label = None


    def RelU(self,x):
        return np.maximum(0,x)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1) 

    def cross_entropy_loss(self, label, prediction):
        epsilon = 1e-10  # Small constant
        self.label = label
        return -np.sum(label*np.log(prediction + epsilon))

    
    def forward(self, input):

        self.input = input

        #first layer
        self.a_hidden_1 = self.RelU(np.dot(input, self.W1) + self.b1)

        #second layer
        self.a_hidden_2 = self.RelU(np.dot(self.a_hidden_1, self.W2) + self.b2)
        
        #output layer
        self.a_output = self.softmax(np.dot(self.a_hidden_2, self.W3) + self.b3)
        
        return self.a_output

    
    def backwards(self):
        
        self.l_output = (self.a_output - self.label)

        self.l_hidden_2 = np.dot(self.W3, self.l_output.reshape(-1,1)) * (self.a_hidden_2>0).reshape(-1,1)
        
        self.l_hidden_1 = np.dot(self.W2, self.l_hidden_2) * (self.a_hidden_1>0).reshape(-1,1)
        
    
    def optimize(self):
        self.W3 -= self.lr * np.outer(self.a_hidden_2, self.l_output)
        self.b3 -= self.lr * self.l_output

        self.W2 -= self.lr * np.outer(self.a_hidden_1, self.l_hidden_2)
        self.b2 -= self.lr * self.l_hidden_2.reshape(1,-1)

        self.W1 -= self.lr * np.outer(self.input, self.l_hidden_1)
        self.b1 -= self.lr * self.l_hidden_1.reshape(1,-1)


    def predict(self, input):
        unmaxed = self.forward(input)
        maxed = np.where(unmaxed == np.max(unmaxed), 1, 0)
        
        return maxed 