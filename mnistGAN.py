"""
author: cs-dan
adapted from: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
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
from keras.datasets.mnist import load_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#
#   Globals
#

###
### DATA PROCESSING AND ACCESSORIES
###

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
    
    (inputsTrain, targetsTrain), (inputsTest, targetsTest) = load_data()
    print(f'Training set shape: {inputsTrain.shape, targetsTrain.shape} \nTesting set shape: {inputsTest.shape, targetsTest.shape}')
    plt.figure('Dataset Sampleset ( 5x5 )')
    SamplePlot(25, inputsTrain)
    return

#
#   function: DataGenReal
#   Gens actual samples via random
#
def DataGenReal(dataset, SampleNum):
    
    index = np.random.randint(0, dataset.shape[0], SampleNum)
    inputs = dataset[index]
    targets = np.ones((SampleNum, 1))
    return inputs, targets

#
#   function: DataGenFake
#   Gens nonreal samples, via uniform rand num between 0 and 1
#
def DataGenFake(SampleNum):

    inputs = np.random.rand( 28 * 28 * SampleNum )
    inputs = inputs.reshape(( SampleNum,28,28,1 ))
    targets = np.zeros(( SampleNum,1 ))
    return inputs, targets

#
#   function: DataPreproc
#   Loads inputs, expands to 3d from 2d, 
#   converts to appropriate type and scales for binary crossentropy
#
def DataPreproc():

    (inputsTrain, _), (_, _) = load_data()
    expansion = np.expand_dims(inputsTrain, axis=-1)
    expansion = expansion.astype('float32') / 255.0
    return expansion

###
### DISCRIMINATOR AND ACCESSORIES
###

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
#   function: DiscriminatorTrain
#   Train on discriminator
#
def DiscriminatorTrain(model, dataset, IterNum, BatchNum):

    for i in range(IterNum):
        inputsTrue, targetsTrue = DataGenReal(dataset, int(BatchNum / 2))
        inputsFalse, targetsFalse = DataGenFake(int(BatchNum / 2))
        lossTrue, accuracyTrue = model.train_on_batch(inputsTrue, targetsTrue)
        lossFalse, accuracyFalse = model.train_on_batch(inputsFalse, targetsFalse)
        print(f'@ iteration {i} - Real image accuracy: {accuracyTrue*100:.2f}% \tFake image accuracy: {accuracyFalse*100:.2f}% \tLoss on True set: {lossTrue:.2f} \tLoss on False set: {lossFalse:.2f}')
    return model

#
#   function: DisModelSetup
#   Literally the name
#
def DisModelSetup():

    model = DiscriminatorSetup(( 28,28,1 ))
    print(model.summary())
    #utils.plot_model(model, to_file='Discriminator-Plot.png', show_shapes=True, show_layer_names=True)
    return model

#
#   function: DisModelTrain
#   Revs up that discriminator
#
def DisModelTrain(model):
    
    dataset = DataPreproc()
    model = DiscriminatorTrain(model, dataset, 100, 256)
    return

###
### GENERATOR AND ACCESSORIES
###

#
#   function: GenLatent
#   Generates the points in latent space to use as input
#
def GenLatent(dims, numInputs):

    X = np.random.randn(dims, numInputs)
    X = X.reshape(numInputs, dims)
    return X

#
#   function: GeneratorGenFake
#   Gens fake x and y 
#
def GeneratorGenFake(model, dims, numInputs):
    X = GenLatent(dims, numInputs)
    Y = np.zeros((numInputs, 1))
    X = model.predict(X)
    return X, Y

#
#   function: GeneratorSetup
#   Defines the Gen
#
def GeneratorSetup(latentDims):

    model = keras.Sequential([
        #   base 7x7
        layers.Dense(128 * ( 7*7 ), input_dim=latentDims),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape(( 7,7,128 )),
        #   up to 14x14
        layers.Conv2DTranspose(128, ( 4,4 ), strides=( 2,2 ), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        #   up to 28x28
        layers.Conv2DTranspose(128, ( 4,4 ), strides=( 2,2 ), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, ( 7,7 ), activation='sigmoid', padding='same')
    ])
    return model

#
#   function: GenModelSetup
#   Setups the gen and prints info about the structure
#
def GenModelSetup():

    LatentDims = 100
    numInputs = 25
    model = GeneratorSetup(LatentDims)
    print(model.summary())
    #utils.plot_model(model, to_file='Discriminator-Plot.png', show_shapes=True, show_layer_names=True)
    X, _ = GeneratorGenFake(model, LatentDims, numInputs)
    SamplePlot(numInputs, X)
    return model

###
### GAN MODEL   (Not a model itself, but a collection of both)
###

#
#   function: GANsetup
#   Sets up the GAN model
#   

###
### MAIN SCRIPT
###

#
#   function: main
#   Runs the whole thing   
#
def main():
    
    """data phase"""
    DataLoad()

    """model (discriminator) phase"""
    # model = DisModelSetup()
    # DisModelTrain(model)
    
    """model (generator) phase"""
    model = GenModelSetup()
    #GeneratorTrain(model)
    
    #ModelTest()
    #ModelPersist()

    return 1

#
#   Start
#
if __name__ == '__main__':
    
    main()