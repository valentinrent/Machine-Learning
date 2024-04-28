import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim

# # Transform to convert images to PyTorch tensors and normalize the data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Load MNIST dataset
# trainset = datasets.MNIST(root='C:\datasets', train=True, download=True, transform=transform)
# testset = datasets.MNIST(root='C:\datasets', train=False, download=True, transform=transform)

# # DataLoader
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# # Visualizing samples of the digits
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # Function to show images
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     plt.imshow(img.numpy().squeeze(), cmap='gray_r')

# # Plotting some examples
# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     imshow(images[index])

# plt.show()



# Define the Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

# Initialize the model
model = Net()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the Model
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)  # Flatten MNIST images into a 784 long vector
        optimizer.zero_grad()  # Zero the gradients
        output = model(images)  # Pass batch
        loss = criterion(output, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Add up the loss
    print(f"Training loss: {running_loss/len(trainloader)}")

# Test the Model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# Save the Trained Model
torch.save(model.state_dict(), 'mnist_model.pth')

# The saved