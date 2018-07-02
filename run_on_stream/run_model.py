import torch
import sys
import os
import time
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import Tkinter as tk
from skimage import io
from PIL import ImageTk, Image

# NN Class definition
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

def callback():
	path = "rock.jpg"
	img = ImageTk.PhotoImage(Image.open(path))
	panel.configure(image=img)
	panel.image = img

# Play a round
def play():
	image_paths = ["rock.jpg", "paper.jpg", "scissors.jpg", "nothing.jpg"]
	s, image = cam.read()
	if(s):
		image = cv2.resize(image, (32, 32))
		transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		image = transform(image).detach().numpy()
		X = np.zeros([1, 3, 32, 32])
		X[0] = image
		X = X.astype(np.float32)
		X = torch.from_numpy(X)
		y_guess = np.argmax(net(X).detach().numpy())
		guess = classes[y_guess]
		if(guess != "nothing"):
			if(game_strategy == 0):
				# Always win
				path = image_paths[(y_guess+1)%3]
				print(classes[(y_guess+1)%3])
			elif(game_strategy == 1):
				# Always lose
				path = image_paths[(y_guess-1)%3]
				print(classes[(y_guess-1)%3])
			else:
				# Always tie
				path = image_paths[y_guess]
				print(classes[y_guess])
		else:
			path = image_paths[3]
		img = ImageTk.PhotoImage(Image.open(path))
		panel.configure(image=img)
		panel.image = img
	window.after(100, play)

# Set Game Strategy
if(len(sys.argv) != 2):
	sys.exit()
else:
	game_strategy = int(sys.argv[1])

# Set up neural net 
net = torch.load("model.pt")
classes = ("rock", "paper", "scissors", "nothing")
cam = cv2.VideoCapture(1)

# Set up GUI
window = tk.Tk()
window.title("Rock Paper Scissors")
window.geometry("400x400")
path = "nothing.jpg"
img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
window.after(2000, play)
window.mainloop()
