import os
from shutil import copyfile

train_data_portion = .8
train_metadata = []
test_metadata = []
counter = 0
for label, elem in enumerate(["rock_images", "paper_images", "scissors_images", "nothing_images"]):
	img_num = float(len(os.listdir(elem)))
	for i, image_name in enumerate(os.listdir(elem)):
		if(i/img_num < train_data_portion):
			copyfile(elem + "/" + image_name, "../train/" + str(counter) + ".jpg")
			train_metadata.append([str(counter) + ".jpg", label])
			counter += 1
		else:
			copyfile(elem + "/" + image_name, "../test/" + str(counter) + ".jpg")
			test_metadata.append([str(counter) + ".jpg", label])
			counter += 1

f = open("../train/metadata.txt", "w")
for elem in train_metadata:
	f.write(elem[0] + " " + str(elem[1]) + "\n")
f.close()

f = open("../test/metadata.txt", "w")
for elem in test_metadata:
	f.write(elem[0] + " " + str(elem[1]) + "\n")
f.close()
