#!/bin/bash

# Remove old data
rm rock_images -r
rm paper_images -r
rm scissors_images -r
rm ../test -r
rm ../train -r

# Create directories for new images
mkdir rock_images
mkdir paper_images
mkdir scissors_images
mkdir ../test
mkdir ../train

# Generate new images from video
./vid_to_images.sh

# Move and label data to test and train directories
python move_data.py
