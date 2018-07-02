#!/bin/bash

ffmpeg -ss 00:00:02 -t 00:00:30 -f v4l2 -framerate 25 -video_size 640x480 -i /dev/video1 $1
