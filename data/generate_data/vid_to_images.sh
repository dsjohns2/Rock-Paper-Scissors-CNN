#!/bin/bash

# Create new images from video
ffmpeg -i rock.mp4 -ss 00:00:02 -r 10 ./rock_images/$filename%03d.jpg
ffmpeg -i paper.mp4 -ss 00:00:02 -r 10 ./paper_images/$filename%03d.jpg
ffmpeg -i scissors.mp4 -ss 00:00:02 -r 10 ./scissors_images/$filename%03d.jpg
ffmpeg -i nothing.mp4 -ss 00:00:02 -r 10 ./nothing_images/$filename%03d.jpg

# Resize new images
for path in "rock_images/" "paper_images/" "scissors_images/" "nothing_images/";
do
	for image in $( ls $path);
	do
		image_path="$path$image"
		convert $image_path -resize 32x32\! $image_path
	done
done
