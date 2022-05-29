# -*- coding: utf-8 -*-

"""
FileName:               wGANModels
Author Name:            Arun M Saranathan
Description:            This code file contains the models for the generator and discriminator models that I plan to use
                        for the Wasserstein DCGANs. This set of models is designed for hyperspectral data

Date Created:           15th January 2020
Last Modified:          22nd January 2020
"""

'Import keras functionalities'
from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


'THE GENERATOR MODELS'
class generatorsGANModels():
    def __init__(self, spectraDim=240, dropout=0.4, genFilters=400, filterSize=11, input_dim=50):
        """
        The constructor which helps you decide the model the model properties. In this set of models I'm going to create
        models that are pyramidal in shape

        :param spectraDim: The length of the spectrum for which the model is being built (Default = 240)
        :param dropout: The rate of dropout included in my code (Default = 0.4)
        :param genFilters: The number of filters in the first layer of the generator (Default = 400)
        :param filterSize: The size of the convolutional filter (Default = 11)
        :param inputDim: Size of the latent vector (Default = 50)
        """
        self.spectraDim = spectraDim
        self.dropout = dropout
        self.genFilters = genFilters
        self.filterSize = filterSize
        self.input_dim = input_dim


    def ganModel_Gen1(self):
        """
        This function creates a generator of an architecture with 5 Convolutional layers.

        :return: The Keras generator models
        """

        'Input Layer'
        'In: 100 X 1, depth =1'
        'Out: 15 X 1, depth = 400'
        generator = Sequential()
        generator.add(Dense(15 * self.genFilters, input_dim=self.input_dim, kernel_initializer='he_normal'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Reshape((15, self.genFilters)))

        'Convolutional Layer-1'
        'In: 15 X 1, depth =400'
        'Out: 30 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 2), self.filterSize, padding='same', kernel_initializer='he_normal'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(LeakyReLU())

        'Convolutional Layer-2'
        'In: 30 X 1, depth = 200'
        'Out: 60 X 1, depth = 100'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 4), self.filterSize, padding='same', kernel_initializer='he_normal'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(LeakyReLU())

        'Convolutional Layer-3'
        'In: 60 X 1, depth = 100'
        'Out: 120 X 1, depth = 50'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 8), self.filterSize, padding='same', kernel_initializer='he_normal'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(LeakyReLU())

        'Convolutional Layer-4'
        'In: 120 X 1, depth = 50'
        'Out: 240 X 1, depth = 25'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 16), self.filterSize, padding='same', kernel_initializer='he_normal'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(LeakyReLU())

        'Convolutional Layer-5'
        'In: 240 X 1, depth = 25'
        'Out: 240 X 1, depth = 1'
        generator.add(Conv1D(1, self.filterSize, padding='same', kernel_initializer='he_normal'))
        generator.add(Activation('tanh'))
        #generator.add(Flatten())

        return generator


'THE DISCRIMINATOR MODELS'
class discriminatorsGANModels():
    def __init__(self, spectraDim=240, dropout=0.4, disFilters=25, filterSize=11, input_dim=50):
        """
        The constructor which helps you decide the model the model properties. In this set of models I'm going to create
        models that are pyramidal in shape

        :param spectraDim: The length of the spectrum for which the model is being built (Default = 240)
        :param dropout: The rate of dropout included in my code (Default = 0.4)
        :param genFilters: The number of filters in the first layer of the generator (Default = 400)
        :param filterSize: The size of the convolutional filter (Default = 11)
        :param inputDim: Size of the latent vector (Default = 50)
        """
        self.spectraDim = spectraDim
        self.dropout = dropout
        self.disFilters = disFilters
        self.filterSize = filterSize
        self.input_dim = input_dim

    def ganModels_Dis1(self, ganType="JSD"):
        """
        This function creates a generator of an architecture with 4 Convolutional layers and 1 Dense decision layer.

        :param ganType: This flag tells us whether the discriminator is for a conventional JSD GAN or a wGAN critic
                     (Default =  'JSD')

        :return: The Keras generator models
        """

        discriminator = Sequential()
        'LAYER -1'
        'In: 240 X 1, depth = 1'
        'Out: 120 X 1, depth = 25'
        #discriminator.add(Reshape((self.spectraDim, 1), input_shape=(self.spectraDim,)))
        discriminator.add(Conv1D(filters=self.disFilters, kernel_size=self.filterSize, strides=2, padding='same',
                                 input_shape=(self.spectraDim,1,), kernel_initializer= 'he_normal'))
        discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(self.dropout))

        'LAYER -2'
        'In: 120 X 1, depth = 25'
        'Out: 60 X 1, depth = 50'
        discriminator.add(Conv1D(filters=self.disFilters*2, kernel_size=self.filterSize, strides=2, padding='same',
                                 kernel_initializer='he_normal'))
        discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(self.dropout))

        'LAYER -3'
        'In: 60 X 1, depth = 50'
        'Out: 30 X 1, depth = 100'
        discriminator.add(Conv1D(filters=self.disFilters * 4, kernel_size=self.filterSize, strides=2, padding='same',
                                 kernel_initializer='he_normal'))
        discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(self.dropout))

        'LAYER -4'
        'In: 30 X 1, depth = 100'
        'Out: 15 X 1, depth = 200'
        discriminator.add(Conv1D(filters=self.disFilters * 8, kernel_size=self.filterSize, strides=2, padding='same',
                                 kernel_initializer='he_normal'))
        discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(self.dropout))

        'Output Layer'
        discriminator.add(Flatten())
        discriminator.add(Dense(1, kernel_initializer='he_normal'))
        if(ganType == "JSD"):
            discriminator.add(Activation('sigmoid'))

        return discriminator


'Check model structure'
if __name__ == "__main__":
    'Create the generator'
    gen1 = generatorsGANModels().ganModel_Gen1()
    print(gen1.summary())

    'Create the discriminator'
    dis1 = discriminatorsGANModels().ganModels_Dis1(ganType="JSD")
    print(dis1.summary())

