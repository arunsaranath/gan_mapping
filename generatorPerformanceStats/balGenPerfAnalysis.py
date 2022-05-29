# -*- coding: utf-8 -*-

"""
FileName:               genPerfAnalysis
Author Name:            Arun M Saranathan
Description:            This file analyzes the performance of the generator relative to the MICA data in the continuum
                        removed CRISM spectra with clear absorptions
Date Created:           03rd December 2019
Last Modified:          03rd December 2019
"""
'Import and set up Tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
set_session(tf.Session(config=config))

'Import keras layers and optimizers'
from keras.models import Model, Sequential
from keras.layers import *
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

'General python libraries'
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.metrics.pairwise import cosine_similarity as cosineDist

'import spectral Python'
import spectral.io.envi as envi
'import neural network models'
from ganTraining.GANModel_1d import GANModel_1d

class genPerfAnalysis():
    def __init__(self, micaLibName='/Volume1/data/CRISM/arun/UMass_redMICA_CR_enhanced.sli'):
        try:
            if not micaLibName:
                raise IOError
            else:
                'Extract the MICA spectra'
                micaHdrName = micaLibName + '.hdr'
                'Read in the spectra'
                crmicaSLI = envi.open(micaHdrName, micaLibName)
                crmica_data = crmicaSLI.spectra
                #self.crmica_data = crmica_data[:, 4:244]
                self.crmica_data = crmica_data
                header = envi.read_envi_header(micaHdrName)
                'Extract the wavelength'
                wvl = header['wavelength']
                wvl = np.asarray(wvl, dtype='f4')
                self.wvl = wvl[4:244]
                self.spectraName = np.asarray(header['spectra names'])

        except IOError:
            self.wvl = np.asarray([])
            self.crmica_data = np.asarray([])
            self.spectraName = np.asarray([])
            print('MICA Library file is needed')


    def initGANModel(self, nBands=240, nGenFilters=250, nDisFilters=20, filterSize=11, genWeightsLoc='',
                     disWeightsLoc=''):
        """
        This function will create and networks for a specific shape and loads the generators and discriminator weights
        if they have been provided.
        ----------------------------------------------------------------------------------------------------------------
        :param nBands: Size of the final layer for the generator and first layer of the discriminator.
        :param nGenFilters: Assuming upper pyramid number of filters in the first layer of the generator.
        :param nDisFilters: Assuming lower pyramid number of filters in the first layer of the discriminator.
        :param filterSize: The size of the filters in terms of number of bands.
        :param genWeightsLoc: The location where the generator weights are present.
        :param disWeightLoc: The location where the generator weights are present.
        :return:
        """

        if self.wvl.size == 0:
            print('MICA Library file is needed')
            return  np.asarray([]),  np.asarray([])

        'Create the object for the specific GAN architecture'
        obj = GANModel_1d(img_rows=nBands, dropout=0.0, genFilters=nGenFilters, disFilters=nDisFilters,
                          filterSize=filterSize)
        'Create the generator for this model'
        gen1 = obj.genModel_CV_L6s2()
        if genWeightsLoc:
            gen1.load_weights(genWeightsLoc)

        'Create the discriminator for this model'
        dis1 = obj.disModel_CV_L6s2()
        if disWeightsLoc:
            dis1.load_weights(disWeightsLoc)

        self.gen1 = gen1
        self.nGenFilters = nGenFilters
        self.dis1 = dis1
        self.nDisFilters = nDisFilters
        self.filterSize = filterSize
        self.nBands = nBands

    def createFeatureModel(self, nDisFilters=''):
        """
        This function will generate a feature extractor with corresponding to the discriminator stored in the object
        ----------------------------------------------------------------------------------------------------------------
        :param dis1: The discmrinator from which we have to extract the features
        :param nBands: Size of the first layer of the discriminator.
        :param nDisFilters: Assuming lower pyramid number of filters in the first layer of the discriminator.
        :param filterSize: The size of the filters in terms of number of bands.
        :return:
        """
        nDisFilters = self.nDisFilters
        model_l2 = Sequential()
        'Layer-1'
        'In: 240 X 1, depth =1'
        'Out:120 X 1, depth =25'
        model_l2.add(Conv1D(filters=nDisFilters, kernel_size=self.filterSize, strides=2, input_shape=(self.nBands, 1),
                     weights=self.dis1.layers[0].get_weights(), activation='relu', padding='same'))

        # model_l2.add(Dropout(0.5))
        'Layer-2'
        'In: 120 X 1, depth =25'
        'Out: 60 X 1, depth =50'
        model_l2.add(Conv1D(filters=nDisFilters * 2, kernel_size=self.filterSize, strides=2,
                            weights=self.dis1.layers[1].get_weights(), padding='same', activation='relu'))

        'Layer-3'
        'In: 60 X 1, depth =50'
        'Out: 30 X 1, depth =100'
        model_l2.add(Conv1D(filters=4*nDisFilters, kernel_size=self.filterSize, strides=2,
                            weights=self.dis1.layers[2].get_weights(), padding='same', activation='relu'))

        'Layer-4'
        'In: 30 X 1, depth =100'
        'Out: 15 X 1, depth =200'
        model_l2.add(Conv1D(filters=8*nDisFilters, kernel_size=self.filterSize, strides=2,
                            weights=self.dis1.layers[3].get_weights(), padding='same', activation='relu'))
        model_l2.add(Flatten())
        model_l2.compile(loss='binary_crossentropy', optimizer=Adam())

        'Extract MICA activations for this model'
        'Reshape to pass on to discriminator'
        crmica_data = self.crmica_data.reshape(self.crmica_data.shape[0], self.crmica_data.shape[1], 1)
        'Get the predictions at output layer'
        self.mica_dataPreds_l2 = model_l2.predict(crmica_data)
        self.model_l2 = model_l2

        return model_l2

    def bestGuessOnSpectra(self, sampleMat):
        """
        This function calculate the cosine similarity between the activations and thresholds out all values under 0.707
        as well as values which are not highest for the sample
        :param sampleMat: The matrix which contains the samples of interest
        :param model_l2: The feature extractor
        :return:
        """
        sampleMat =  sampleMat.reshape( sampleMat.shape[0],  sampleMat.shape[1], 1)
        'Predict activations on validation set'
        sampleMat_Preds = self.model_l2.predict(sampleMat)
        'Find the cosine distance between the exemplars and the data'
        dist = np.squeeze(cosineDist(self.mica_dataPreds_l2, sampleMat_Preds))
        'Remove all values below cos(pi/4)'
        #dist[dist < np.cos(np.pi / 4)] = 0
        dist = dist.T
        'Only preserve the endmember with the highest value'
        #dist2 = dist * (np.argsort(np.argsort(dist)) >= (dist.shape[1] - 1))

        return dist

    def plot_grouped_by_mica(self, sampleMat, similarity):
        """
        This function will take the similarity matrix and generate the plot for each MICA class with he closest
        endmembers for that specific class

        :param sampleMat:
        :param similarity:
        :return:
        """
        'Find the best guess mineral'
        matches = np.argmax(similarity, axis=1)

        'Group spectra by their matches'
        groups={}
        for idx, match in enumerate(matches):
            if match not in groups:
                groups[match] = set([(idx, similarity[idx][match])])
            else:
                groups[match].add((idx, similarity[idx][match] ))

        'For each class in generated spectrs:'
        'plot spectra and save plot with filename as mica match'
        for endmember, sample_idxs in groups.items():
            n=0
            avg_sim = 0.

            for idx, sim in list(sample_idxs):
                if sim >= 0.85:
                    plt.plot(self.wvl, sampleMat[idx, :], 'r--', alpha=(sim+1.)/2.)
                    avg_sim += sim
                    n += 1

            if n>=1:
                avg_sim/=float(n)
                plt.plot(self.wvl, self.crmica_data[endmember], 'k-')
                plt.title('{:s}: avg_sim={:1.1f}'.format(self.spectraName[endmember], avg_sim), fontsize=18)
                plt.xlabel('Wavelength', fontsize=24)
                plt.ylabel('CR CRISM I/F', fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=18)
                filename=''.join([c for c in self.spectraName[endmember] if c.isalpha() or c.isdigit()]).rstrip()
                plt.savefig(('/Volume2/arunFiles/python_HSITools/generatorPerformanceStats/balGenPerf/MICASpectra/'+'mica_{}_{}.png'.format(endmember, filename)))
                plt.close()

    def plot_training_hist(self, similarity):
        """
        This function will take the similarity matrix produce a histogram of the best guesses

        :param similarity:
        :return:
        """

        'Threshold out values beloow 0.707'
        similarity[similarity < np.cos(np.pi /4.)]
        'Remove rows that only contain 0'
        similarity = similarity[~np.all(similarity==0, axis=1)]

        'Find the best guess mineral and group the'
        matches = np.argmax(similarity, axis=1)

        num_matches = len(set(list(matches)))

        fig, ax = plt.subplots()
        bins = np.arange(0, len(list(self.spectraName))+1)
        n, _, _ = plt.hist(matches, bins=bins, density=True, color='b', alpha=1)
        ax.set_ylabel('counts')
        ax.set_title('Training Endmember Histogram ({:1.0f}k samples, {:d}/{:d} classes)'.
                     format(matches.shape[0]/1000, num_matches, len(list(self.spectraName))))
        ax.set_xticks(np.arange(0, len(list(self.spectraName))))
        labels=[]
        for i, name in enumerate(list(self.spectraName)):
            labels += ['('+str(np.sum(matches==i))+')' + name]
        ax.set_xticklabels(labels, rotation=90)
        fig.tight_layout
        plt.savefig(('/Volume2/arunFiles/python_HSITools/generatorPerformanceStats/balGenPerf/MICASpectra/'+'trainHist.png'),
                    bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    '------------------------------------------------------------------------------------------------------------------'
    'SETUP MODELS'
    '------------------------------------------------------------------------------------------------------------------'
    'Create the models'
    sliName = '/Volume2/arunFiles/python_HSITools/crismBalancingDatasets/dataProducts/mica_CRendmembers_reduced.sli'
    obj1 =  genPerfAnalysis(micaLibName=sliName)
    'The weights locations are'
    gen1Loc = '/Volume2/arunFiles/python_HSITools/trainedModels/balancedDataSets/Models/Model-1_big_25e-6_19Dec'+\
              '/generator/gen_cR_75.h5'
    dis1Loc = '/Volume2/arunFiles/python_HSITools/trainedModels/balancedDataSets/Models/Model-1_big_25e-6_19Dec'+\
              '/discriminator/dis_cR_75.h5'
    'Get the models for this architecture'
    obj1.initGANModel(nGenFilters=500, nDisFilters=25, genWeightsLoc=gen1Loc, disWeightsLoc=dis1Loc)
    'Get the Feature Extractor'
    obj1.createFeatureModel()

    '------------------------------------------------------------------------------------------------------------------'
    'SAMPLE GENERATOR DISTIBUTION AND EXTRACT FEATURES'
    '------------------------------------------------------------------------------------------------------------------'
    print('Starting Generation\n')
    'Generate spectra from the generator'
    try_input = np.random.rand(500000, 50)
    genSpectra = np.squeeze(obj1.gen1.predict(try_input))


    '------------------------------------------------------------------------------------------------------------------'
    'EXTRACT BEST GUESS'
    '------------------------------------------------------------------------------------------------------------------'
    print('Performing Similarity Analysis\n')
    'Find the cosine distance between the exemplars and the data'
    similarity = obj1.bestGuessOnSpectra(genSpectra)

    obj1.plot_training_hist(similarity)
    obj1.plot_grouped_by_mica(genSpectra, similarity)







