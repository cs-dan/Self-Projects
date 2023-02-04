"""
author: cs-dan
adapted from: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
"""
"""
    We're using the cifar100 dataset this time.
    With 100 classes.
    "We GAN"
"""
#
#   Dependencies and globals
#
from tensorflow import keras
from keras import layers
from keras import utils
from keras import optimizers
from keras.datasets.cifar100 import load_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#
#   Data Section
#

#
#   function: SamplePlot
#   Loads and preprocesses the dataset
#
def SamplePlot(size, dataset):

    for i in range(size):
        plt.subplot(10, 10, 1+i )
        plt.axis('off')
        plt.imshow(dataset[i], cmap='gray_r')
    plt.show()
    return

#
#   function: DataProc
#   Loads and preprocesses the dataset
#
def DataProc():

    (inputsTrain, targetsTrain), (inputsTest, targetsTest) = load_data()
    print(f'Training set shape: {inputsTrain.shape, targetsTrain.shape} \tTesting set shape: {inputsTest.shape, targetsTest.shape}')
    plt.figure('Dataset Sampleset ( 10x10 )')
    SamplePlot(100, inputsTrain)
    return

#
#   Model Section
#

#
#   function: ModelSetup()
#   Setups the Discriminator (in this case a CNN), prints a summary of the infrastructure
#
def ModelSetup():
    return

#
#   Script Section
#
def main():
    
    """ Data Preproc """
    DataProc()
    ModelSetup()

if __name__ == '__main__':
    main()