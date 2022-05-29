from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


"""config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))"""


hdrName = '/Volume1/data/CRISM/yuki/legacy/sabcondv4/jezero/HRL000040FF/HRL000040FF_07_IF183L_TRR3_atcr_sabcondv4_1_Lib11123_1_4_5_l1_gadmm_a_v2_ca_ice_b200_nr.hdr'
hdr = envi.read_envi_header(hdrName)

if 'wavelength' in hdr:
    try:
        wavelength = [float(b) for b in hdr['wavelength']]
    except:
        pass
wavelength = wavelength[4:244]

del hdrName, hdr

class GANModel_Train():
    def __init__(self, generator, discriminator, location, tableName, dataStore='', learningRate=1e-5, verbose=True,
                 noiseSize=50, batch_size=256, epochs=50, decayRate=0.9):
        'Model parameters & hyperparameters'
        self.gen = generator
        self.dis = discriminator
        self.location = location
        self.tableName = tableName
        self.learningRate = learningRate
        self.verbose = verbose
        self.noiseSize = noiseSize
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_count = 2000000 // self.batch_size
        self.dataStore = dataStore
        self.decayRate = decayRate

        'if the locations dont exist create them'
        if not os.path.isdir(self.location):
            os.makedirs(self.location)
            os.makedirs(self.location + '/generator')
            os.makedirs(self.location + '/discriminator')
            os.makedirs(self.location + '/images')
            os.makedirs(self.location + '/disLogs')
            os.makedirs(self.location + '/ganLogs')

        'Noise input for tracking model progress'
        self.try_input = np.random.rand(25, self.noiseSize)

        'define the model parameters'
        self.dis.compile(loss=['binary_crossentropy'],
                         optimizer=Adam(lr=self.learningRate), metrics=['accuracy'])

        'MAKE GAN-1 as combination of Generator 1-Discriminator 1'
        self.dis.trainable = False
        ganInput = Input(shape=(self.noiseSize,))
        'getting the output of the generator'
        'and then feeding it to the discriminator'
        'new model = D(G(input))'
        x = self.gen(ganInput)
        ganOutput = self.dis(x)
        self.gan = Model(inputs=ganInput, outputs=ganOutput)
        self.gan.compile(loss=['binary_crossentropy'],
                         optimizer=Adam(lr=self.learningRate), metrics=['accuracy'])

        if self.verbose:
            self.gen.summary()
            self.dis.summary()
            self.gan.summary()

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    # function to train from a HDF5 Store
    def GAN_trainFromStore(self):
        'Check if a store is given as input'
        if not self.dataStore:
            print('Cannot call GAN_trainFromStore() without providing a store to train from')
            return

        'Extract batch for training from store'
        for e in range(self.epochs):
            for j in tqdm(range(self.batch_count)):
                with pd.HDFStore(self.dataStore, mode='r') as newstore:
                    df_restored = newstore.select(self.tableName, start=(j * self.batch_size),
                                                  stop=(self.batch_size * (j + 1)))
                    x_batch = np.asarray(df_restored)
                    x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], 1)

                # Use generator and predict
                noise_input1 = np.random.rand(self.batch_size, self.noiseSize)
                pred1 = self.gen.predict(noise_input1, batch_size=self.batch_size)
                # pred1 = np.squeeze(pred1)
                # Make Data and Labels for Discriminator
                x_discriminator = np.concatenate([pred1, x_batch])
                y_discriminator = [0] * self.batch_size + [1] * self.batch_size

                # Train the discriminator
                self.dis.trainable = True
                d_loss1 = self.dis.train_on_batch(x_discriminator, y_discriminator)
                # bNum = e

                # Train the Generator
                noise_input2 = np.random.rand(2*self.batch_size, self.noiseSize)
                y_generator = [1] * (self.batch_size*2)

                self.dis.trainable = False
                g_loss1 = self.gan.train_on_batch(noise_input2, y_generator)

            # plot the progress
            print("Model-2 %d [D loss: %f][G loss: %f]\n" % (e, d_loss1[0], g_loss1[0]))
            # print(self.location)

            # if e % 5 == 0:
            self.save_imgs(epoch=e)

            # create a tensorboard log
            tb1 = TensorBoard(log_dir=(self.location + '/disLogs'), histogram_freq=0, write_graph=True, write_images=False)
            tb1.set_model(self.dis)
            self.write_log(tb1, ['loss', 'acc'], d_loss1, e)

            # create a tensorboard log
            tb2 = TensorBoard(log_dir=(self.location + '/ganLogs'), histogram_freq=0, write_graph=True,
                              write_images=False)
            tb2.set_model(self.gan)
            self.write_log(tb2, ['loss', 'acc'], g_loss1, e)

    # Function to save inputs and outputs
    def save_imgs(self, epoch):
        r, c = 5, 5
        preds = self.gen.predict(self.try_input)
        preds = np.squeeze(preds)
        fig = plt.figure(figsize=(10, 10))
        for i in range(preds.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.plot(wavelength, np.squeeze(preds[i, :]), linewidth=2.0)
            plt.tight_layout()

        # Save the figure
        fig.savefig(self.location + ("images/samples_generated_%d.png" % epoch))
        plt.close('all')
        # Save the generator weights
        self.gen.save_weights(self.location + ("generator/gen_cR_%d.h5" % epoch))
        # Save the discriminator weights
        self.dis.save_weights(self.location + ("discriminator/dis_cR_%d.h5" % epoch))

