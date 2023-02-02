"""
Wanted to learn how to make a GAN
"""
"""
For self:
    "the point"
    GANS are for generating, you're trying to create something.
    So if the purpose is creative, you're on the right track for use.
    
    "gonna need a few things"
    A discriminator - CNN for classifying if input is real or generated.
    A generator - Does the opposite to transform input to a full 2d image.
    
    "goals"
    How to make and train the discriminator.
    How to make the generator and train both disc and genr in conjunction.
    How to eval GAN performance and use generator to generate images.

"""
"""
Planned setup: Procedural
"""
#
#   Dependencies
#
from tensorflow import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#
#   Globals
#

#
#   function: SamplePlot
#   Plots from dataset
#
def SamplePlot(size, dataset):
    
    for i in range(size):
        plt.subplot(5, 5, 1+i )
        plt.axis('off')
        plt.imshow(dataset[i], cmap='gray_r')
    plt.show()
    return

#
#   function: DataLoad
#   Loads the data, prints out shape and first 25 samples
#
def DataLoad():
    
    from keras.datasets.mnist import load_data
    (inputsTrain, targetsTrain), (inputsTest, targetsTest) = load_data()
    print(f'Training set shape: {inputsTrain.shape, targetsTrain.shape} \nTesting set shape: {inputsTest.shape, targetsTest.shape}')
    plt.figure('Dataset Sampleset ( 5x5 )')
    SamplePlot(25, inputsTrain)
    return

#
#   function: main
#   Runs the whole thing   
#
def main():
    
    """stuff goes here"""
    DataLoad()
    #DataPreproc()
    #ModelSetup()
    #ModelLoad()
    #ModelTrain()
    #ModelTest()
    #ModelPersist()

#
#   Start
#
if __name__ == '__main__':
    
    main()