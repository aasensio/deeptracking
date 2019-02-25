# Tracking of solar structures using deep learning

## Introduction

This repository presents a deep neural network that predicts the velocity
field in the xy plane from a pair of consecutive frames. The neural network
is a deep convolutional neural network that takes two consecutive frames
as input. The outputs are the maps of displacement vectors (vx,vy) which, 
when applied to the first image, gives the second image as output. This is
fundamentally the optical flow from image 1 to image 2 in the couple of
images. The network is trained in the following manner:

- A convolutional neural network (CNN) predicts the two maps vx and vy 
from the pair of input images.
- A spatial transformer (bilinear interpolation) is used to warp the
first image applying the predicted optical flow.
- The same spatial transformer is used to warp the second image by
applying the negative optical flow.
- The resulting warped first image is compared with an L1 Charbonnier
loss with the second image. The same is done with the other image.
- A smooth loss is added to the vx and vy maps to force smooth velocity
fields.

## Retraining 

The networks can be retrained using `train.py` provided some input files with
the observed images are provided. They are currently hardwired in the code.
New training files can be generated with the `gen_db.py` file. This program
reads FITS files from all available channels, extracts random patches
from the files and writes an HDF5 file with the training and validation sets.

## Predicting

Predictions of velocity field maps can be done with `test.py`. The file
defines a class that reads the trained network. The class can
generate a movie (with the `movie` method) and can also do a single
frame prediction (the `test` method). This method does the following:
- Reads two consecutive frames.
- Makes sure that the size is a multiple of 8 (requisite of the network
so that the output has the same dimensions as the input).
- Compute the maximum and minimum of the images and normalize by them
so that the images are in the [0,1] range.
- Applies the network, giving the flow in the two directions plus
the warped outputs.


## Dependencies

- numpy
- h5py
- astropy
- tqdm
- pytorch
- skimage