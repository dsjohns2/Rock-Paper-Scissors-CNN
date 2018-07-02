#!/bin/bash

# Create new images from video
rm images/*
ffmpeg -i random.mp4 -ss 00:00:00 -r 10 ./images/$filename%04d.jpg

# Resize new images
for path in "images/";
do
	for image in $( ls $path);
	do
		image_path="$path$image"
		convert $image_path -resize 32x32\! $image_path
	done
done
