"""
author: cs-dan
"""
"""
Wanted to learn how to make a GAN

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
from keras import utils
from keras import optimizers
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
#   function: DataPrepare
#   Loads inputs, expands to 3d from 2d, 
#   converts to appropriate type and scales for binary crossentropy
#
def DataPrepare():

    (inputsTrain, _) (_, _) = load.data()
    expansion = expand_dims(inputsTrain, axis=-1)
    expansion = expansion.astype('float32') / 255.0
    return expansion

#
#   function: DataGenReal
#   Gens actual samples via random
#
def DataGenReal(dataset, SampleNum):
    
    inputs = dataset[randint(0, dataset.shape[0], SampleNum)]
    targets = ones((SampleNum, 1))
    return inputs, targets

#
#   function: DataGenFake
#   Gens nonreal samples, via uniform rand num between 0 and 1
#
def DataGenFake(SampleNum):

    inputs = rand( 28 * 28 * SampleNum )
    inputs = inputs.reshape(( SampleNum,28,28,1 ))
    targets = zeros(( SampleNum,1 ))
    return inputs, targets

#
#   function: DataPreproc
#   Loads input from training, 
#   converts 2d into 3d for CNN input,
#   scales vals to 0 and 1
#
def DataPreproc():



#
#   function: DiscriminatorSetup
#   Setup the discriminator portion
#
def DiscriminatorSetup(shape):
    model = keras.Sequential([
        layers.Conv2D(64, ( 3,3 ), strides=( 2,2 ), padding='same', input_shape=shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(64, ( 3,3 ), strides=( 2,2 ), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=2e-4, beta_1=0.5), metrics=['accuracy'])
    return model

#
#   function: ModelSetup
#   Literally the name
#
def ModelSetup():
    model = DiscriminatorSetup((28,28,1))
    print(model.summary())
    utils.plot_model(model, to_file='Discriminator-Plot.png', show_shapes=True, show_layer_names=True)

#
#   function: main
#   Runs the whole thing   
#
def main():
    
    """stuff goes here"""
    DataLoad()
    DataPreproc()
    ModelSetup()
    #ModelLoad()
    #ModelTrain()
    #ModelTest()
    #ModelPersist()
    return 1

#
#   Start
#
if __name__ == '__main__':
    
    main()