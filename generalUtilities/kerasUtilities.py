# -*- coding: utf-8 -*-

"""
FileName:               kerasUtilities
Author Name:            Arun M Saranathan
Description:            This code file certain keras Utility files that are required for a variety of operations like
                        Wasserstein GANs etc..

Date Created:           15th January 2020
Last Modified:          9th November 2020
"""
'Import default python libraries needed'
import numpy as np
import tensorflow as tf
'Import keras functionalities'
from tensorflow.keras import backend as K

class kerasUtilities():
    def wassersteinLoss(self, y_true, y_pred):
        """
        This function implements the loss needed for the Wasserstein GAN.

        The Wasserstein loss function is very simple to calculate. In a standard GAN. the discriminator has a sigmoid
        output, representing the probability that samples are real or generated. In Wasserstein GANs, however, the
        output is linear with no activation function. Instead of being constrained to [0, 1], the discriminator wants
        to make the distance between its output for the real and generated samples as large as possible.

        The most natural way to acheive this is to label generated samples -1 and real samples 1, instead of 0 and 1 as
        is used in JSD-GANs, so that multiplying the outputs by the labels will give you the loss immediately.

        Note that the nature of this loss means that it can be (and frequently will be) less than 0

        Original Code Source: github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

        :param y_true: The true labels with real data as 1 and fake data as -1
        :param y_pred: The values predicted by the wGAN - critic.

        :return: The loss value calculated by the model
        """

        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
        """
        Calculates the geadient penalty loass for a batch of "averaged" samples.

        In Improved wGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function that penalizes
        the network if the gradient moves away from 1. However, it is impossible to evaluate this function at all points
        in the input space,The compromise used in the paper is to choose random points on the lines between real and
        generated samples, and check the gradient at these points. Note that it is the gradient w.r.t. the input
        averaged samples, not the weights of the discriminator that we are penalizing.

        In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss. Then
        we get the gradients of the discriminator w.r.t. the input averaged samples. The l2 norm and penalty can then be
        calculated for this gradient.

        Note that this loss function requires the original averaged samples as input, but Keras only supports passing
        'y_true' and 'y_pred' to loss functions. To get around this , we make a partial() of the function with
        'averaged_samples' argument, and use that for model training.

        References
        ----------------------------------------------------------------------------------------------------------------
        [1] Gulrajani, Ishaan, et al. "Improved training of wasserstein GANs", Advances in neural information processing
        systems, pp. 5767-5777. 2017.

        Original Code Source: github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

        :param y_true:
        :param y_pred:
        :param averaged_samples:
        :param gradient_penalty_weight:
        :return:
        """

        'First set the gradients'
        'Assuming: that y_pred has dimensions (batch_size, 1)'
        '          averaged samples has dimensions (batch_size, nbr_features)'
        'gradients afterwards has dimension (batch_size, nbr_features), basically'
        'a list of nbr_features-dimensional gradient vectors'
        gradients = K.gradients(y_pred, averaged_samples)[0]

        'Compute Euclidean norm by squaring'
        gradients_sqr = K.square(gradients)
        '...... summing over the row...'
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        '.......... and sqrt'
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        'Compute lambda*(1 - ||grad||)^2 still for each single sample'
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        'return the mean as loss over all the batch samples'
        return K.mean(gradient_penalty)

    def decayed_learning_rate(self, step, initial_learning_rate=2e-4, decay_rate=0.99, decay_step=100000, staircase=True):
        """
        This function can be used to implement a decayed learning rate for neural network training.

        :param step: The step for which we are calculating the learning rate
        :param initial_learning_rate: the initial learning rate (Default=2e-4)
        :param decay_rate: The rate at which it is decaying (Default=0.99)
        :param decay_step: scaling the decay rate (Default= 100000)
        :param staircase: does the learning rate have a staircase structure (Default= True)
        :return:
        """
        if not staircase:
            return initial_learning_rate * decay_rate ** ((step / decay_step))
        else:
            return initial_learning_rate * decay_rate ** ((step / decay_step))


    def checkModelParams(self, model, inputSize, outputSize):
        """
        This function will check whether input and output-sizes provided match the model size
        :param model: The keras model of interest
        :param inputSize: The input size provided by the user
        :param outputSize: The output size provided by the user
        :return: nothing (raises an error if the test fails)
        """
        'Get the generator layers'
        gl = []
        for layer in model.layers:
            gl.append([layer.input_shape, layer.output_shape])

        genInputSize, genOutputSize = gl[0][0][1], gl[-1][1][1]

        if ((genInputSize != inputSize) or (genOutputSize != outputSize)):
            try:
                raise Exception('Mod:Mismat', ('The provided details on model input and output' +
                                               'sizes does not match the provided details'))
            except 'Mod:MisMat' as err:
                print(err.args(0) + '-' + err.args(1))



