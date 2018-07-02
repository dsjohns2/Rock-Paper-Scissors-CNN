import torch
import os
import time
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage import io

# Class Definitions
class Net(nn.Module):
	""" Rock Paper Scissors Convolutional Neural Network """
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 4)

	def forward(self, x):
		#print("Start: " + str(x.size()))
		x = F.relu(self.conv1(x))
		#print("after conv1: " + str(x.size()))
		x = self.pool(x)
		#print("after pool1: " + str(x.size()))
		x = F.relu(self.conv2(x))
		#print("after conv2: " + str(x.size()))
		x = self.pool(x)
		#print("after pool2: " + str(x.size()))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# Load the neural net
net = torch.load("model.pt")

# Test the neural net
classes = ("rock", "paper", "scissors", "nothing")
image_list = os.listdir("./images")
image_list.sort()
for image_name in image_list:
	image = io.imread("./images/" + image_name)
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	image = transform(image).detach().numpy()
	X = np.zeros([1, 3, 32, 32])
	X[0] = image
	X = X.astype(np.float32)
	X = torch.from_numpy(X)
	y_guess = np.argmax(net(X).detach().numpy())
	print(classes[y_guess])
	time.sleep(.1)
