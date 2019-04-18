# Satellite Image Segmentation
Solution that we have achieved on GIS Tech Hack hackathon on building segmentation

## Hackathon task
The task on that hackathon was to make a mask of buildings on a given image taken from satellite. The image was taken in Russia, Republic of Tatarstan, Kazan, Aviastroitelny district.

The organizers has provided us with servers for computing. These servers have **GPU NVIDIA TESLA V100 with 32GB RAM** on board

The input image size was 6528x7734 pixels with 3 channels (RGB).

## Solution:
1. The original images and train masks were cropped to the image size 256x256 with 3 channels.
2. The architecture of neural network was similar to U-Net but with some modifications.
    * As encoder part we have taken pretrained ResNet50 convolution network on *imagenet* dataset
    * Decoder model have 5 upsampling layers. On each layer (exept the last) there is 2 convolution layers with batch normaliztion
3. You can download weights [here](https://yadi.sk/d/9OkyfadDlntJjA)    

## Results
![Imgur](https://i.imgur.com/A3IznSR.png)
