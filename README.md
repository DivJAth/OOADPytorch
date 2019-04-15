# OOADPytorch
This repo demonstrates the underlying Object Oriented Design of PyTorch.
We train a Convolution Neural Network on the MNIST dataset.

## Basic MNIST Example

![sampleMNIST](IMAGES/sampleMNIST.png)

This project implements a beginner classification task on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset with a [Convolutional Neural Network(CNN or ConvNet)](https://en.wikipedia.org/wiki/Convolutional_neural_network) model. 

The entire code is written in Pytorch
<p align="center"><img width="20%" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" /></p>

## Requirements
 - Package requirements: Torch, Torchvision
 - Will work on only python 3
 
## Usage
 `./main.py --kwargs
 
 kwargs:
 - *batch_size* [default`=64]
 - *test_batch_size* [default=100]
 - *epochs* [default=5]
 - *lr* [default=0.05]    //learning rate
 - *momentum* [default=0.5]
 - *no_cuda* [default=False]
 - *log_interval* [default=100]
 - *save_model* [default=True]
 
## Example
`./main.py --batch_size=32 --lr=0.01

 
 
