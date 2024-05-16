# %% [markdown]
# # From Multilayer Perceptrons to Convolutional Neural Networks

# %% [markdown]
# In this notebook, we will start with a simple Multilayer Perceptron (MLP) and then gradually build up to a Convolutional Neural Network (CNN) to classify the FashionMNIST and CIFAR10 datasets. We will use PyTorch to build and train the models.

# %% [markdown]
# ## Preliminaries

# %%
# Libraries to be imported, please make sure you have them installed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from torchvision import datasets, transforms
from tqdm.autonotebook import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context



# %% [markdown]
# In PyTorch, we can use the GPU to accelerate the training process. We can check if a GPU is available and set the device accordingly. This will allow us to move the data and the model to the GPU. We check if a GPU is available and set the device accordingly. We also set the random seed for reproducibility.
# 
# 

# %%
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using {device} device")

# Random seed for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# In the following code cells, we define the training and evaluation functions for the models. They follow the same structure as in the previous notebook. However, there are some changes including the addition of metrics to measure the performance and the use of progress bars to track the training progress. They assist in monitoring the training process and help in debugging if the model is not learning as expected. 
# 
# For the progress bars, we use the `tqdm` library. You can install it using `!pip install tqdm`. You can read more about the library [here](https://github.com/tqdm/tqdm). The `tqdm` library provides a simple way to add progress bars to your loops. You can wrap any iterable with `tqdm(iterable)` to display a progress bar that updates in real time. In our case we wrap the `train_loader` and `test_loader` with `tqdm` to display the progress bars during training and testing. The progress bars iterate over the batches in the dataset and display the loss and the metrics for each batch.
# 
# For the metrics, we use the `torchmetrics` library. You can install it using `!pip install torchmetrics`. You can read more about the library [here](https://torchmetrics.readthedocs.io/en/latest/). After each training epoch, as well as after testing the model on the test dataset, we want to output the following metrics for the model:
# - Accuracy
# - F1 score
# 
# The Accuracy is the ratio of the number of correct predictions to the total number of predictions. It is defined as:
# 
# $$
# \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
# $$
# 
# The F1 score is the harmonic mean of the precision and recall. It is defined as:
# 
# $$
# \text{F1 score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
# $$
# 
# where Precision is the ratio of the number of true positive predictions to the total number of positive predictions. It is defined as:
# 
# $$
# \text{Precision} = \frac{\text{Number of true positive predictions}}{\text{Total number of positive predictions}}
# $$
# 
# and Recall is the ratio of the number of true positive predictions to the total number of actual positive samples. It is defined as:
# 
# $$
# \text{Recall} = \frac{\text{Number of true positive predictions}}{\text{Total number of actual positive samples}}
# $$
# 
# 

# %%
def train_model(model, train_loader, optimizer, loss_fn, epochs, device="cpu"):
    model = model.to(device)  # Move the model to the device
    model.train()  # Set the model to training mode

    # ToDo: Initialize the metrics
    numclasses = len(train_loader.dataset.classes)
    print(numclasses)
    acc = Accuracy(task='multiclass', num_classes=numclasses).to(device)
    f1 = F1Score(task='multiclass',num_classes=numclasses).to(device)

    running_loss = []  # Initialize the running loss

    progress_bar1 = tqdm(
        range(epochs),
        desc="Epochs [Loss: -,  Acc: -, AUROC: -, F1: -]",
        position=0,
        leave=True,
    )  # Progress bar for epochs

    for _ in progress_bar1:
        progress_bar2 = tqdm(
            train_loader, desc="Loss: -", position=1, leave=True
        )  # Progress bar for batches

        for x, y in progress_bar2:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # ToDo: Forward pass
            f_x = model(x)
            loss = loss_fn(f_x, y)

            # ToDo: Backward pass
            loss.backward()

            optimizer.step()

            # ToDo: Update the metrics
            acc.update(f_x, y)
            f1.update(f_x, y)

            running_loss.append(loss.item())  # Update the running loss
            progress_bar2.set_description(desc=f"Loss: {running_loss[-1]:.3f}")

        avg_loss = sum(running_loss) / len(running_loss)
        running_loss = []
        progress_bar1.set_description(
            desc=f"Epochs [Loss: {avg_loss:.3f},  Acc: {acc.compute():.3f}, F1: {f1.compute():.3f}]"
        )  # Update the progress bar description for epochs

# %%
@torch.no_grad()  # Decorator to disable gradient calculation
def evaluate_model(model, test_loader, loss_fn, device="cpu"):
    model = model.to(device)  # Move the model to the device
    model.eval()  # Set the model to evaluation mode

    # ToDo: Initialize the metrics
    numclasses = len(test_loader.dataset.classes)
    acc = Accuracy(task='multiclass', num_classes=numclasses).to(device)
    f1 = F1Score(task='multiclass',num_classes=numclasses).to(device)

    running_loss = []  # Initialize the running loss

    progress_bar = tqdm(test_loader, desc="Loss: -", leave=True)

    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)

        # ToDo: Forward pass
        f_x = model(x)
        loss = loss_fn(f_x, y)

        # ToDo: Update the metrics
        acc.update(f_x, y)
        f1.update(f_x, y)

        running_loss.append(loss.item())  # Update the running loss
        progress_bar.set_description(desc=f"Loss: {running_loss[-1]:.3f}")

    avg_loss = sum(running_loss) / len(running_loss)
    print(
        f"Evaluation Results: [Loss: {avg_loss:.3f},  Acc: {acc.compute():.3f}, F1: {f1.compute():.3f}]"
    )

# %% [markdown]
# We now load the FashionMNIST and CIFAR10 datasets using the `torchvision` library. It is a good practice to normalize the dataset by calculating the mean and standard deviation of the dataset and then using them to normalize the dataset. The mean and standard deviation for both the FashionMNIST and CIFAR10 datasets are provided below:
# 
# - FashionMNIST:
#     - Mean: 0.2860
#     - Standard Deviation: 0.3530
# - CIFAR10:
#     - Mean: (0.4914, 0.4822, 0.4465)
#     - Standard Deviation: (0.2023, 0.1994, 0.2010)
# 
# Also, since our model expects the input to be tenors, we convert the images and labels to tensors. To convert the images to tensors and to normalize the dataset, we use the `transforms` module from the `torchvision` library. We use the `Compose` class to chain the transformations together. We first convert the images to tensors and then normalize the dataset using the mean and standard deviation of the dataset. We then pass the `Compose` object to the `transform` argument of the `Dataset` class to apply the transformations to the dataset.
# 
# After loading the dataset, we create the data loaders using the `DataLoader` class. We pass the dataset and the batch size to the `DataLoader` class to create the data loaders. The batch size is a hyperparameter that determines the number of samples that will be passed through the model at once. A larger batch size can speed up the training process, but it requires more memory. A smaller batch size can slow down the training process, but it requires less memory. We set the batch size to 64 for both the FashionMNIST and CIFAR10 datasets, but you are free to experiment with different batch sizes.

# %%
batch_size = 64  # Batch size for training and testing

# %%
# FMNIST

# Image transformations
transform = transforms.Compose(
    [
        # ToDo: Add image transformations
        # 1) Convert images to tensors
        # 2) Normalize the dataset
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
        
    ]
)

# ToDo: Download the training and test datasets
fmnist_train_dataset = datasets.FashionMNIST(root="C:\datasets", train=True, transform=transform, download=True)
fmnist_test_dataset =  datasets.FashionMNIST(root="C:\datasets", train=False, transform=transform, download=True)

# ToDo: Prepare the data loaders
fmnist_train_loader = torch.utils.data.DataLoader(fmnist_train_dataset, batch_size=batch_size, shuffle=True)
fmnist_test_loader = torch.utils.data.DataLoader(fmnist_test_dataset, batch_size=batch_size, shuffle=False)

# %%
# CIFAR-10

# Image transformations
transform = transforms.Compose(
    [
        # 1) Convert images to tensors
        # 2) Normalize the dataset
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ]
)

# ToDo: Download the training and test datasets
cifar10_train_dataset = datasets.CIFAR10(root="C:\datasets", train=True, transform=transform, download=True)
cifar10_test_dataset = datasets.CIFAR10(root="C:\datasets", train=False, transform=transform, download=True)

# ToDo: Prepare the data loaders
cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# Since all our tasks involve image classification, we will use the same loss function for all the models. We will use the Cross Entropy Loss function, which is commonly used for classification tasks. It is defined as:
# 
# $$
# L(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)
# $$
# 
# where $y_i$ is the true label and $\hat{y}_i$ is the predicted label.

# %%
# ToDo: Define Loss function
loss_fn = nn.CrossEntropyLoss()

# %% [markdown]
# We use the Stochastic Gradient Descent (SGD) and Adam ([link to paper](https://arxiv.org/abs/1412.6980)) optimizers in this notebook, but you can experiment with other optimizers like AdaGrad, RMSprop, etc. The learning rate is a hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function. In the SGD algorithm the learning $\alpha$ rate is defined as: 
# 
# $$
# \phi_{t+1} = \phi_t - \alpha \nabla L(f(x_t; \phi_t), y_t)
# $$
# 
# where $\phi_t$ are the current weights, $\phi_{t+1}$ are the updated weights, $\nabla L(f(x_t; \phi_t), y_t)$ is the gradient of the loss function with respect to $\phi_t$, $f(x_t; \phi_t)$ is the prediction of the model for $x_t$, and $y_t$ is the true label of $x_t$.
# 
# We set the learning rate to 0.001 for all the models, but you are free to experiment with different learning rates.

# %%
learning_rate = 1e-3  # Learning rate for the optimizer

# %% [markdown]
# One complete pass through the dataset is called an epoch. We train the model for a certain number of epochs. We set the number of epochs to 30 for all the models, but you are free to experiment with different numbers of epochs.

# %%
epochs = 30 # Number of epochs

# %% [markdown]
# ## FashionMNIST with MLP

# %% [markdown]
# We define the MLP model for the FashionMNIST dataset. The model consists of **two hidden layers with ReLU activation functions and a final output layer with a softmax activation function**. The hyperparameters of the hidden layers are defined as follows:
# - Hidden Layer 1: width = 128
# - Hidden Layer 2: width = 64
# 
# > Please consider that in PyTorch the softmax function is already applied in the CrossEntropy loss function.
# 
# We then initialize the model and the optimizer. In this case, we use the Stochastic Gradient Descent (SGD) with default parameters and our defined learning rate. 

# %%
# Define the MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # ToDo:  Define the layers of the MLP
            # ...
            # Softmax is included in the loss function !
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            
        )

    def forward(self, x):
        # ToDo: Forward pass
        x = x.view(-1, 28*28)
        x = self.layers(x)
        return x

        

model = MLP()

# Print the model architecture
print(model)

# ToDo: Define  Optimizer using the model parameters and the learning rate
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# %%
# # Run the entire pipeline
train_model(model, fmnist_train_loader, optimizer, loss_fn, epochs=epochs)
evaluate_model(model, fmnist_test_loader, loss_fn)

# %% [markdown]
# ## CIFAR10 with MLP

# %% [markdown]
# Before we can train the MLP model on the CIFAR10 dataset, we need to modify the MLP model to accept the input shape of the CIFAR10 dataset. The CIFAR10 dataset consists of color images with a shape of (3, 32, 32). We need to modify the model to accept this input shape. Also, due to the high dimensionality of the CIFAR10 dataset, we need to increase the number of hidden layers and also the number of neurons in the hidden layers to enable the model to learn more complex patterns in the data. Our new model will consist of **three hidden layers with ReLU activation functions and a final output layer with a softmax activation function**. The hyperparameters of the hidden layers are defined as follows:
# - Hidden Layer 1: width = 256
# - Hidden Layer 2: width = 128
# - Hidden Layer 3: width = 64
# 
# > Please consider that in PyTorch the softmax function is already applied in the CrossEntropy loss function.
# 
# We then instantiate the model and define the optimizer. We use the same optimizer and learning rate as before.

# %%
# Define the MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # ToDo:  Define the layers of the MLP
            # ...
            # Softmax is included in the loss function !
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # ToDo: Forward pass
        x = x.view(-1, 32*32*3)
        x = self.layers(x)
        return x


model = MLP()

# Print the model architecture
print(model)

# ToDo: Define Optimizer using the model parameters and the learning rate
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# %%
# # Run the entire pipeline
train_model(model, cifar10_train_loader, optimizer, loss_fn, device=device, epochs=epochs)
evaluate_model(model, cifar10_test_loader, loss_fn, device=device)

# %% [markdown]
# ### Exercises
# 
# After training the MLP model on the datasets, please answer the following questions:
# 
# 1. What are the train- and test-accuracies of the MLP models?
# 2. What are the train- and test-F1-scores of the MLP models?
# 3. Does the MLP model generalize well on the datasets? Explain your answer.
# 4. How does the MLP model perform on the CIFAR10 dataset compared to the FashionMNIST dataset?
# 5. What are some possible explanations for differences in performance?
# 6. Can we use the MLP model for datasets with higher resolution images? Explain your answer.

# %% [markdown]
# ## Results
# 1. 
#     * MNIST: 
#     1. Train: 79,4%
#     3. Test: 83,4%
#     * CIFAR10:
#     1. Train: 39.1%
#     3. Test: 47.8%
# 2. 
#     * MNIST: 
#     1. Train: 0.794
#     3. Test: 0.835
#     * CIFAR10:
#     1. Train: 0.391
#     3. Test: 0.478
# 3. Generalization seems to be good since test data perfoms slightly better than training data
# 4. The MLP performes approx. two times worse on the CIFAR10 dataset compared to FashionMNIST.
# 5. The CIFAR10 dataset contains more complex images and features such as color are not considered.
# 6. While it is possible, complexity will get very high making MLP's ineffecive 
# 

# %% [markdown]
# ## CIFAR10 with Simple CNN

# %% [markdown]
# In this section we introduce a simple Convolutional Neural Network (CNN) model for the CIFAR10 dataset. The model consists of **two convolutional layers with ReLU activation functions, followed by a max pooling layer, one fully connected layer with ReLU activation function, and a final output layer with a softmax activation function.** The hyperparameters of the hidden layers are defined as follows:
# - Convolutional Layer 1: out_channels = 32, kernel_size = 3
# - Max Pooling Layer: kernel_size = 2
# - Convolutional Layer 2: out_channels = 64, kernel_size = 3
# - Max Pooling Layer: kernel_size = 2
# - Fully Connected Layer: width = 64
# 
# > Please consider that in PyTorch the softmax function is already applied in the CrossEntropy loss function.
# 
# We then instantiate the model and define the optimizer. In this case, we use the Adam optimizer with default parameters and our defined learning rate.

# %%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # ToDo: Define the layers of the CNN model
            # ...
            # Softmax is included in the loss function !
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        # ToDo: Forward pass
        # ...
        x = self.layers(x)
        return x


model = CNN()

# Print the model architecture
print(model)

# ToDo: Define Optimizer using the model parameters and the learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# # Run the entire pipeline
train_model(model, cifar10_train_loader, optimizer, loss_fn, device=device, epochs=epochs)
evaluate_model(model, cifar10_test_loader, loss_fn, device=device)

# %% [markdown]
# ### Exercises
# 
# After training the simple CNN model on the CIFAR10 dataset, please answer the following questions:
# 1. What are the train- and test-accuracies of the CNN models?
# 2. What are the train- and test-F1-scores of the CNN models?
# 3. Does the simple CNN model generalize well on the CIFAR10 dataset? Explain your answer.
# 4. What improvements can be made to the simple CNN model to improve its generalization performance?
# 5. How does the simple CNN model perform on the CIFAR10 dataset compared to the MLP model?
# 6. What are the advantages of using a CNN model over an MLP model for image classification tasks?
# 7. What results do you expect if we use the CNN model for the FashionMNIST dataset?
# 

# %% [markdown]
# ## Results
# 1. 
#     * CIFAR10:
#     1. Train: 89.7%
#     3. Test: 68.7%
# 2. 
#     * CIFAR10:
#     1. Train: 0.897
#     3. Test: 0.687
# 3. In this case the model does not seem to generalize well since training accuracy and F1 score are much higher than when using test data.
# 4. Methods such as Regularization and Batch Normalization can usually improve generalization.
# 5. The CNN model performes almost twice as good as the MLP model
# 6. CNN models take the sourroundings of pixels into consideration and can better detect features. Furthermore they require less parameters and are easier to train.
# 7. Similar or better results
# 
# 

# %% [markdown]
# ## CIFAR10 with Modern CNN

# %% [markdown]
# In the previous sections, we worked with neural networks that are considered shallow by today's standards. In this final section we introduce a modern CNN model that is widely used for image classification tasks, namely the **ResNet18** model from the ResNet family. The ResNet18 model consists of **18 layers with residual connections**. You can read more about the ResNet18 model [here](https://arxiv.org/abs/1512.03385).
# 
# The purpose of this section is to demonstrate the superior performance of deeper CNN models over shallow CNN models. We will train the ResNet18 model on the CIFAR10 dataset and compare its performance with the simple CNN model.
# 
# The question may arise why we don't just add layers to our previous model until it is "deep enough". In general, while defining our own CNN model is a good exercise, using a pre-trained model like ResNet18 can save time and computational resources. For example, we can use the pre-trained model as a feature extractor and then train a small fully connected layer on top of it. This iknown as transfer learning, and it is a common technique used in practice. However, transfer learning is out of the scope of this notebook. Here, we will train the ResNet18 model from scratch. 
# 
# Instantiating the model is simple in PyTorch. It exists as predefined class in the `torchvision.models` module as well as in the `torch.hub` module. However, since the ResNet18 model was built for the ImageNet dataset, which consists of color images with a resolution of 224x224, we need to modify the model to accept the input shape of the CIFAR10 dataset. Furthermore, each image in ImageNet has 1000 classes, so we need to modify the output layer to have 10 classes for the CIFAR10 dataset.
# 
# We then define the optimizer.

# %%
model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", weights=None)

# Modify the first layer to accept 3 x 32 x 32 images
# Hint: You can access the model layers and modify them
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# Modify the last layer to have 10 classes
# Hint: You can access the model layers and modify them
model.fc = nn.Linear(512, 10)

# Print the model architecture
print(model)

# ToDo: Define Optimizer using the model parameters and the learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# Run the entire pipeline
train_model(model, cifar10_train_loader, optimizer, loss_fn, device=device, epochs=epochs)
evaluate_model(model, cifar10_test_loader, loss_fn, device=device)

# %% [markdown]
# ### Exercises
# 
# After training the ResNet18 model on the CIFAR10 dataset, please answer the following questions:
# 1. What is the train- and test-accuracy of the ResNet18 model?
# 2. What is the train- and test-F1-score of the ResNet18 model?
# 3. Does the ResNet18 model generalize well on the CIFAR10 dataset? Explain your answer.
# 4. How does the ResNet18 model perform on the CIFAR10 dataset compared to the simple CNN model?
# 5. How many parameters does the ResNet18 model have compared to the simple CNN model? Is the difference reflected in the performance? How about the training time?
# 6. Can we expect the same behavior if we created a deeper MLP model instead of using the ResNet18 model? Explain your answer.
# 

# %% [markdown]
# ## Results
# 1. 
#     * ResNet18:
#     1. Train: 94,5%
#     3. Test: 81.3%
# 2. 
#     * ResNet18:
#     1. Train: 0.945
#     3. Test: 0.813
# 3. While not perfect it definetely generalizes better than the simple CNN model and since overall acurracy is fairly high it is acceptable.
# 4. Both training and testing show an increase of approx. 10% in acurracy. 
# 5. ResNet18 consists of 11 million parameters while our CNN has approx. 300.000. Even though we see an increase in accuracy it is not proportional to the amount of parameters and more importantly to the training time.
# 6. While deeper networks might further improve performance MLP's lack the ability to easily understand spatial relationships making it nearly impossible to outperfome CNN's.
# 
# 
# 


