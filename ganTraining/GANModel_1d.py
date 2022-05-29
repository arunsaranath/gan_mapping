# -*- coding: utf-8 -*-

"""
FileName:               GANModel_1d
Author Name:            Arun M Saranathan
Description:            This file includes implementation of different models for the generator and discriminator that
                        we use in our model. We use keras models

Date Created:           05th December 2017
Last Modified:          03rd September 2019
"""

from keras.models import Model, Sequential
from keras.layers import *
#from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.optimizers import Adam


class GANModel_1d():
    """
    --------------------------------------------------------------------------------------------------------------------
    FUNCTION NAME INTERPRETATION
    --------------------------------------------------------------------------------------------------------------------
    xxxModel -> the first three characters describe the kind of model 'gen' for
    generators and 'dis' for discriminators
    _XX -> the next two charcaters denote the type of connection 'FC' for fully
    connected and 'CV' for convolutional
    _Lysy -> L denotes layers and s the stride size, therefore 'L6s2' denotes 6
    layers with stride.'L2s2_L6s1' denotes 2 layers with stride 2 followed by 4
    layers with stride 1

     All padding is 'same'
     dropout = 0.4
     Batch normalization is applied
     Activation is 'relu'
    --------------------------------------------------------------------------------------------------------------------
    """

    'The Constructor'
    def __init__(self, img_rows=240, dropout=0.4, genFilters=400,
                 disFilters=25, filterSize=5, input_dim=50):
        self.img_rows = img_rows
        self.dropout = dropout
        self.genFilters = genFilters
        self.disFilters = disFilters
        self.filterSize = filterSize
        self.input_dim = input_dim

    def genModel_CV_L6s2(self):
        """
        # GENERATOR-1
        # 6 Layers
        # Upsampling Factor 2 per layer (except first and last)
        # Final Layer Shape: img_rows X 1 X 1
        # Activation: 'relu'
        # bias_initializer = Constant value of 0.1
        :return: keras generator model
        """
        'Convolutional Layer 1'
        'In: 50 X 1, depth =1'
        'Out: 15 X 400, depth = 25'
        generator = Sequential()
        generator.add(Dense(15 * self.genFilters, input_dim=self.input_dim,
                            activation='relu', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Reshape((15, self.genFilters)))

        'LAYER -2'
        'In: 15 X 1 X1, depth = 400'
        'Out: 30 X 1 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 2), self.filterSize,
                             activation='relu', padding='same', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))

        'LAYER -3'
        'In: 30 X 1 X1, depth = 400'
        'Out: 60 X 1 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 4), self.filterSize,
                             activation='relu', padding='same', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))

        'LAYER -4'
        'In: 60 X 1 X1, depth = 400'
        'Out: 120 X 1 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 8), self.filterSize,
                             activation='relu', padding='same', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))

        'LAYER -5'
        'In: 120 X 1 X1, depth = 400'
        'Out: 240 X 1 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 16), self.filterSize,
                             activation='relu', padding='same', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))

        'OUTPUT LAYER'
        'In: 240 X 1 X 1, depth=25'
        'Out: 240 X 1 X 1, depth =1'
        generator.add(Conv1D(1, self.filterSize, padding='same',
                             bias_initializer=Constant(0.1)))
        # generator.add(Flatten())
        generator.add(Activation('sigmoid'))

        return generator

    def disModel_CV_L6s2(self):
        """
        # DISCRIMINATOR-1
        # 5 Layers
        # Downsampling Factor (Stride) 2 per layer (except last)
        # Output Size  = 1 X 1
        # Activation: 'relu'
        # bias_initializer = Constant value of 0.1

        :return: Returns a Keras model with 5 layers, 4 Convolutional and 1 FC
        """

        discriminator = Sequential()
        # 'LAYER -1'
        # 'In: 240 X 1 X 1, depth =1'
        # 'Out: 120 X 1 X 1, depth = 25'
        discriminator.add(Conv1D(filters=self.disFilters, kernel_size=self.filterSize, strides=2,
                                 input_shape=(self.img_rows, 1), bias_initializer=Constant(0.1),
                                 activation='relu', padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # 'LAYER -2'
        # 'In: 120 X 1 X 1, depth =25'
        # 'Out: 60 X 1 X 1, depth = 50'
        discriminator.add(Conv1D(filters=self.disFilters * 2,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # 'LAYER -3'
        # 'In: 60 X 1 X 1, depth =50'
        # 'Out: 30 X 1 X 1, depth = 75'
        discriminator.add(Conv1D(filters=self.disFilters * 4,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # 'LAYER -4'
        # 'In: 30 X 1 X 1, depth =50'
        # 'Out: 15 X 1 X 1, depth = 100'
        discriminator.add(Conv1D(filters=self.disFilters * 8,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # Output Layer
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator

    def disModel_CV_L6s2_old(self):
        """
        # DISCRIMINATOR-1
        # 5 Layers
        # Downsampling Factor (Stride) 2 per layer (except last)
        # Output Size  = 1 X 1
        # Activation: 'relu'
        # bias_initializer = Constant value of 0.1

        Added batchNormalization and dropout

        :return: Returns a Keras model with 5 layers, 4 Convolutional and 1 FC
        """

        discriminator = Sequential()
        # 'LAYER -1'
        # 'In: 240 X 1 X 1, depth =1'
        # 'Out: 120 X 1 X 1, depth = 25'
        discriminator.add(Conv1D(filters=self.disFilters,
                                 kernel_size=self.filterSize, strides=2,
                                 input_shape=(self.img_rows, 1), bias_initializer=Constant(0.1),
                                 activation='relu', padding='same'))

        # 'LAYER -2'
        # 'In: 120 X 1 X 1, depth =25'
        # 'Out: 60 X 1 X 1, depth = 50'
        discriminator.add(Conv1D(filters=self.disFilters * 2,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))

        # 'LAYER -3'
        # 'In: 60 X 1 X 1, depth =50'
        # 'Out: 30 X 1 X 1, depth = 75'
        discriminator.add(Conv1D(filters=self.disFilters * 4,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))

        # 'LAYER -4'
        # 'In: 30 X 1 X 1, depth =50'
        # 'Out: 15 X 1 X 1, depth = 100'
        discriminator.add(Conv1D(filters=self.disFilters * 8,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))

        # Output Layer
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator



    def genModel_CV_L4s2(self):
        """
        # GENERATOR-2
        # 4 Layers
        # Upsampling Factor 2 per layer (except first and last)
        # Final Layer Shape: img_rows X 1 X 1
        # Activation: 'relu'
        # bias_initializer = Constant value of 0.1

        :return: Returns a Keras model with 5 layers, 1FC and 3 Convolutional
        """

        'Convolutional Layer 1'
        'In: 50 X 1, depth =1'
        'Out: 60 X 400, depth = 25'
        generator = Sequential()
        generator.add(Dense(60 * self.genFilters, input_dim=self.input_dim,
                            activation='relu', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Reshape((60, self.genFilters)))
        generator.add(Dropout(self.dropout))

        'LAYER -1'
        'In: 60 X 1 X1, depth = 400'
        'Out: 120 X 1 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 8), self.filterSize,
                             activation='relu', padding='same', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Dropout(self.dropout))

        'LAYER -2'
        'In: 120 X 1 X1, depth = 400'
        'Out: 240 X 1 X 1, depth = 200'
        generator.add(UpSampling1D(size=2))
        generator.add(Conv1D(int(self.genFilters / 16), self.filterSize,
                             activation='relu', padding='same', bias_initializer=Constant(0.1)))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Dropout(self.dropout))

        'OUTPUT LAYER'
        'In: 240 X 1 X 1, depth=25'
        'Out: 240 X 1 X 1, depth =1'
        generator.add(Conv1D(1, self.filterSize, padding='same',
                             bias_initializer=Constant(0.1)))
        # generator.add(Flatten())
        generator.add(Activation('sigmoid'))

        return generator



    def disModel_CV_L4s2(self):
        """
        # DISCRIMINATOR-2
        # 3 Layers
        # Downsampling Factor (Stride) 2 per layer (except last)
        # Output Size  = 1 X 1
        # Activation: 'relu'
        # bias_initializer = Constant value of 0.1

        :return:
        """

        discriminator = Sequential()
        'LAYER -1'
        'In: 240 X 1 X 1, depth =1'
        'Out: 120 X 1 X 1, depth = 25'
        discriminator.add(Conv1D(filters=self.disFilters,
                                 kernel_size=self.filterSize, strides=2,
                                 input_shape=(self.img_rows, 1), bias_initializer=Constant(0.1),
                                 activation='relu', padding='same'))
        discriminator.add(Dropout(self.dropout))

        'LAYER -2'
        'In: 120 X 1 X 1, depth =25'
        'Out: 60 X 1 X 1, depth = 50'
        discriminator.add(Conv1D(filters=self.disFilters * 2,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        discriminator.add(Dropout(self.dropout))

        'LAYER -3'
        'In: 60 X 1 X 1, depth =50'
        'Out: 30 X 1 X 1, depth = 75'
        discriminator.add(Conv1D(filters=self.disFilters * 3,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        discriminator.add(Dropout(self.dropout))

        'Output Layer'
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator

    def disModel_CV_L6s2_rep(self, initModel=''):
        """
        This function creates a discriminator model which creates the final representation

        :param initModel: The model from which the weights are to be extracted if any
        :return:
        """

        if not initModel:
            model_l2 = Sequential()
            model_l2.add(Conv1D(filters=20, kernel_size=11, strides=2, input_shape=(240, 1),
                                padding='same', activation='relu'))
            model_l2.add(Conv1D(filters=40, kernel_size=11, strides=2,
                                padding='same', activation='relu'))
            model_l2.add(Conv1D(filters=80, kernel_size=11, strides=2,
                                padding='same', activation='relu'))
            model_l2.add(Conv1D(filters=160, kernel_size=11, strides=2,
                                padding='same', activation='relu'))
            model_l2.add(Flatten())
            model_l2.compile(loss='binary_crossentropy', optimizer=Adam())
        else:
            model_l2 = Sequential()
            model_l2.add(Conv1D(filters=20, kernel_size=11, strides=2, input_shape=(240, 1),
                                weights=initModel.layers[0].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Conv1D(filters=40, kernel_size=11, strides=2,
                                weights=initModel.layers[2].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Conv1D(filters=80, kernel_size=11, strides=2,
                                weights=initModel.layers[4].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Conv1D(filters=160, kernel_size=11, strides=2,
                                weights=initModel.layers[6].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Flatten())
            model_l2.compile(loss='binary_crossentropy', optimizer=Adam())

        return model_l2

'Check model structure'
if __name__ == "__main__":
    'Create an ojbect of this class'
    obj = GANModel_1d(img_rows=240, dropout=0.0, genFilters=250, disFilters=20,
                      filterSize=11)
    dis1 = obj.disModel_CV_L6s2()
    'Get the pre-trained weights'
    dis1.load_weights(
        '/Volume2/arunFiles/pythonCodeFiles/CRISM_repLearning/modelsOnYukiModel_cR_wGAN/Models/Model-4_small/discriminator/dis_cR_75.h5')
    disRep = obj.disModel_CV_L6s2_rep(dis1)
    disRep.summary()

