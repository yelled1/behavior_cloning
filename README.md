# behavior_cloning
Udacity self driving car behavior cloning project3 in a video game like simulator

## Project has following python package dependencies ##
argparse base64 json numpy  socketio eventlet PIL flask pandas matplotlib scipy keras tensorflow

## Project Codes ##
1. I have added my own helper utility: hDrvUtils.p
which has file IO and image resizing script, as well as randomized camera angle fetcher with steering coefficient compensator.

The above script is provided along with project required:
2. model.py - The script used to create and train the model.
3. drive.py - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version. Slightly modified to take in 64x64 image
4. model.json - The model architecture.
5. model.h5   - The model weights.
* The json & h5 are trained model saved from model.py
The trained model managed to go through the train track w/o being off course.
The data used were 300MB or so provided by udacity & combined with 100MB of my own twice around the track.
No particular part of track were emphasized (which was recommended by some other students in the FAQ).
This & other techniques like change in lighting, sheering, rotating, etc... suggested might improve the poor showing on the validation track, where it just hits a portion of dark wall & do NOT go anywhere after first 20 sec.
* This will be worked on for future reference.

The ML model followed nvidia End to End Learning for Self-Driving Cars https://arxiv.org/pdf/1604.07316v1.pdf
"""
Train the weights of our network to min( MSE ) the steering command output by the network & 
the command of either the human driver, or the adjusted steering command for off-center and rotated images 
9 layers, including a normalization layer, 5 convolutional layers & 3 fully connected layers.

* 1.L image normalization hardcoded using x/127.5 -1 (64x64x3)
* 2. ~4L 2x2 stride & 5x5 kernel CNN - starts out with Filters [24, 36, 48, 64, 64]
* 5. ,6L non-strided CNN 3x3 Kernel
* 7. ~9L fully conn = output inverse turning radius. [1164, 100, 40, 10]
* And, optimized with Adam
* It had an additioan Layer of Dropout, but it had to be dropped

I have used 7 epochs with image sized reduced to 64x64 using a rather powerful nVidia GT 1060 graphics card.
Each iteration took about 5 minutes to train
** other Notables **
The data collected using keyboard is erratic as mentioned by others, but data provided is too smooth. Combining both had definite desired effect. Code itself is pretty much self documenting. Not PEP8, but similar to technique I use at work.
