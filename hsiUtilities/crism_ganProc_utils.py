# -*- coding: utf-8 -*-

"""
FileName:               crism_openmax_utils
Author Name:            Arun M Saranathan
Description:            This code file is contains the various utility files required for creating the labeled dataset
                        from the various CRISM images that I am processing.

Date Created:           30th April 2021
Last Modified:          30th April 2021
"""

import numpy as np
import pandas as pd
import os
from spectral.io import envi
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

from hsiUtilities import hsiUtilities
from generalUtilities import generalUtilities
from ganTraining import GANModel_1d

class crism_ganProc_utils:
    def remFlatSpectra(self, crImgFileName, crHdrFileName=None, imgName=None):
        """
        This function is used to extract from each image only those spectra which have some absorption features in the
        continuum removed spectrum.

        :param crImgFileName: (string)
        This is string which contains the address of the continuum removed CRISM image that is to be processed.

        :param crHdrFileName: (string) [Default: None]
        This is string which contains the address of the continuum removed CRISM image header that is to be processed.
        If none is provided it will be assumed the header has the name and same location as the image except with  a
        '.hdr' extension rather than the '.img' format of the images.

        :param imgName: (string) [Default: None]
        This string is the name of the image. If no image name is provided. It is assumed that the image is stored in a
        folder with the images name.

        :return:
        :var imgData: (ndarray: nSamples X 3)
        This variable contains all the information related to the specific pixel. The first column contains the image
        name and the second column contains the row number and the third column has the column number of each chosen
        pixel.

        :var selectedPix: (ndarray: nSamples X nBands)
        This variable contains the spectral data, each row is the spectrum of one the chosen pixels.
        """
        'If not provided generate the header name'
        if crHdrFileName == None:
            'From the image name get the header address'
            crHdrFileName = crImgFileName.replace('.img', '.hdr')

        'Get the image name'
        if imgName == None:
            portions = crImgFileName.split('/')
            imgName = portions[-2]

        'Read in the image'
        crImg = envi.open(crHdrFileName, crImgFileName)
        crImg = crImg.load()

        'Get the MASK name'
        maskImgFileName = crImgFileName.replace('_CR', '_CR_mask')
        maskHdrFileName = maskImgFileName.replace('.img', '.hdr')

        maskImg = envi.open(maskHdrFileName, maskImgFileName)
        maskImg = np.squeeze(maskImg.load())

        'Get the locations of the interesting pixels from the image'
        [row, col] = np.nonzero(maskImg)

        'Get the interesting pixels'
        selectedPix = np.asarray(crImg[row, col, 4:244])

        imgName = np.asarray([imgName] * row.shape[0])

        'Get the image data information'
        imgData = (np.vstack((imgName, row, col))).T

        return imgData, selectedPix

    def prepare_sliFile(self, sliFileName, sliHdrName=None, contRem=False, sclLvl=None):
        """
        This function is used to extract spectra from the ENVI type SLI file and prepare it for use by the application
        of specific operations like continuum removal, scaling the absorption features etc..

        :param sliFileName: (string)
        This is string which contains the address of the ENVI SLI file that is to be processed.

        :param sliHdrName: (string) [Default: None]
        This is string which contains the address of the header associated with the SL file that is to be processed.
        If none is provided it will be assumed the header has the name and same location as the image except with  a
        '.hdr' extension rather than the '.sli' format of the spectral library.

        :param contRem: (Boolean) [Default: False]
        A boolean variable that indicates whether continuum removal needs to be performed on the spectra in the spectral
        library

        :param sclLvl: (0 < float 1) [Default: None]
        A float variable that indicates the size of the largest absorption band in the spectrum. If no value is provided
        (default) no scaling is done. If it is provided the size of the largest absorption is equal to sclLvl. This
        parameter is in (0, 1]

        :return:
        :var spectra: (ndarray: nSamples X nBands)
        This variable contains the spectral data from the spectral library, each row is the spectrum of one the samples
        from the spectral library.
        """

        'If not provided get the default header name'
        if sliHdrName == None:
            sliHdrName = sliFileName.replace('.sli', '.hdr')

        'Open the SLI file and extract the spectra'
        SLI = envi.open(sliHdrName, sliFileName)
        spectra = SLI.spectra
        header = envi.read_envi_header(sliHdrName)

        'If needed perform the continuum removal'
        if contRem:
            'First get the wavelengths'
            wavelength = np.asarray(header['wavelength'], dtype='single')

            co_spectra = np.zeros((spectra.shape))
            for ii in tqdm(range(spectra.shape[0])):
                temp = np.vstack((wavelength, np.squeeze(spectra[ii, :])))
                co_spectra[ii, :] = generalUtilities().convex_hull(temp.T)

            'Ratio to get the continuum removed'
            spectra = spectra / co_spectra

        'If needed scale the size of the absorption features'
        if sclLvl != None:
            assert ((sclLvl > 0) and (sclLvl > 0)), 'The scale level must be in (0, 1]'
            spectra = hsiUtilities().scaleSpectra(spectra, scaleMin=sclLvl)

        return spectra, header

    def create_rep_model(self, modelLoc=None):
        """
        This function accepts the weights corresponding to a specific set of dicriminator weights.

        :param modelLoc: [string]
        This variable contains the string which is the address of the weights of some pre-trained GAN models.

        :return:
        :var disRep (keras model)
        The feature extractor for the appropriate
        """
        'Create an appropriate discirminator model'
        obj = GANModel_1d(img_rows=240, dropout=0.0, genFilters=250, disFilters=20,
                          filterSize=11)
        dis1 = obj.disModel_CV_L6s2()
        'Load the weights to the discriminator model'
        if modelLoc == None:
            modelLoc = os.path.join('/Volume2/arunFiles/pythonCodeFiles/CRISM_repLearning',
                                          'modelsOnYukiModel_cR_wGAN',
                                          'Models/Model-4_small/discriminator/dis_cR_75.h5')

        dis1.load_weights(modelLoc)
        'Create the feature extractor from the discriminator'
        disRep = obj.disModel_CV_L6s2_rep(dis1)

        return disRep

    def check_featExt_input(self, data):
        """
        This function is used to reshape the data to be an appropriate form for the feature extractor.

        :param data: [ndarray: float]
        This is numpy array with float data. The data needs to have a shape 'nSamples X 240 X 1'. If the data is 2D the
        function will expand the matrix and check if the shape is appropriate. If not it will raise an error

        :return:
        """

        if (len(data.shape) == 2):
            data= data.reshape((data.shape[0], data.shape[1], 1))

        assert len(data.shape) == 3, 'Data has too many dimensions'
        assert data.shape[1] == 240, 'The input spectral data must have 240 dimensions'
        assert data.shape[2] == 1, 'The third dimension cannot have more than 1 dimension'

        return data

    def sample_novelClasses(self, storeAdd, endMem_order, labelStart=0, nSamp=1000, testSize=0.15,
                            sclLvl=0.2):
        """
        This function is used to sample known labeled classes to create a dataset from the additional exemplar classes.

        :param storeAdd: (string)
        This string contains the address of the store which contains this data.

        :param labelStart: (int)
        The parameter is the integer value with which the labels start

        :param nSamp: (int)
        The number of samples of each class. If the class contains less than nSamp then all the samples of the classes
        are added.
        :return:
        """

        xTrain = np.asarray([])
        yTrain = np.asarray([])

        with pd.HDFStore(storeAdd, 'r') as newstore:
            #list = newstore.keys()
            #dataList = [item for item in list if 'df_' in item]
            for ii in range(len(endMem_order)):
                item = 'df_' + (endMem_order[ii].split('('))[0]
                df_temp = newstore.select(item)
                print('Table Name:{} with {:d} samples'.format(item, df_temp.shape[0]))

                if df_temp.shape[0] > nSamp:
                    sampSpectra = np.asarray(df_temp.sample(n=nSamp))
                else:
                    sampSpectra = np.asarray(df_temp)
                sampLabels = np.asarray([labelStart] * sampSpectra.shape[0])

                'Scale to the appropriate level'
                sampSpectra = hsiUtilities().scaleSpectra(sampSpectra, scaleMin=sclLvl)

                'Add to output dictionary'
                if xTrain.shape[0] == 0:
                    xTrain = sampSpectra
                    yTrain = sampLabels
                else:
                    xTrain = np.vstack((xTrain, sampSpectra))
                    yTrain = np.hstack((yTrain, sampLabels))

                labelStart += 1


        return (xTrain, yTrain)

    def get_train_test(self, micaDataStore, exemDataStore, endMem_order, srcDataTableName=None, labelTableName=None, sclLvl=0.2,
                       nSamp=1000, testSize=0.2):
        """
        This function can be used create the train and test set with samples from all the known labeled
        classes.

        :param micaDataStore: (String)
        This string is the address of the pandas HDFS store which contains samples corresponding to the MICA classes.

        :param exemplarDataStore: (String)
        This string is the address of the pandas HDFS store which contains samples corresponding to the exemplar
         classes.

        :param srcDataTableName: (String) [Default:None]
        This string is the name of the data table in the micaDataStore. If none is given the function assumes that the
        tablename is 'IF_mixedSamples_CR'

        :param labelTableName: (String) [Default:None]
        (String) [Default:None]
        This string is the name of the label table in the micaDataStore. If none is given the function assumes that the
        tablename is 'Labels_mixedSamples'

        :param sclLvl (0 < float 1) [Default: 0.2]
        A float variable that indicates the size of the largest absorption band in the spectrum. If no value is provided
        (default) no scaling is done. If it is provided the size of the largest absorption is equal to sclLvl. This
        parameter is in (0, 1].

        :param nSamp: (int) [Default: 1000]
        This parameter indicates the number of samples selected for each class

        :param nSamp: (int) [Default: 1000]
        This parameter indicates the number of samples selected for each class

        :return:
        """

        'Set default values if needed'
        if srcDataTableName is None:
            srcDataTableName = 'IF_mixedSamples_CR'

        if labelTableName is None:
            labelTableName = 'Labels_mixedSamples'

        'Create variable to hold the input and the output'
        xTrain = np.asarray([])
        yTrain = np.asarray([])

        'Get the MICA data'
        with pd.HDFStore(micaDataStore, 'r') as srcStore:
            df_x = srcStore.select(srcDataTableName)
            df_y = srcStore.select(labelTableName)

        df_y = df_y.reset_index(drop=True)
        df_x['Labels'] = df_y

        'Now group these by labels'
        df_grp = df_x.groupby('Labels')
        for grp, df in df_grp:
            sampSpectra = np.asarray(df.sample(n=nSamp))
            sampLabels = np.asarray(sampSpectra[:, -1], dtype=int)
            sampSpectra = sampSpectra[:, :-1]

            'Scale to the appropriate level'
            sampSpectra = hsiUtilities().scaleSpectra(sampSpectra, scaleMin=sclLvl)

            'Add to output dictionary'
            if xTrain.shape[0] == 0:
                xTrain = sampSpectra
                yTrain = sampLabels
            else:
                xTrain = np.vstack((xTrain, sampSpectra))
                yTrain = np.hstack((yTrain, sampLabels))


        """fig1 = plt.figure()
        plt.plot(xTrain[5, :])
        plt.plot(xTrain[2005, :])
        plt.show()"""

        lblSTart = yTrain.max() + 1
        xTrain_exm, yTrain_exm = self.sample_novelClasses(exemDataStore,
                                                          endMem_order=endMem_order[lblSTart:],
                                                          labelStart=lblSTart,
                                                          sclLvl=sclLvl,
                                                          nSamp=nSamp)

        'Stack both datasets'
        xTrain = np.vstack((xTrain, xTrain_exm))
        yTrain = np.hstack((yTrain, yTrain_exm))
        yTrain = to_categorical(yTrain)


        'Split the training data into and validation split'
        #x_train, x_val, y_train, y_val = train_test_split(xTrain, yTrain, test_size=testSize)
        return (self.check_featExt_input(xTrain), yTrain)

    def save_model_state(self, model, path, modelName):
        """
        This function is used to save the model state (assumes a keras path)
        :param model: The model which we want to save
        :param path: The folder where we want to save the data
        :param modelName: The name under which the model is stored
        :return:
        """
        if not os.path.exists(os.path.join(os.path.split(path)[0], modelName)):
            os.makedirs(os.path.join(os.path.split(path)[0], modelName))

        'If the architecture has not been saved - save it'
        modelArch = os.path.join(os.path.split(path)[0], modelName, 'modelArch.json')

        model_json = model.to_json()
        with open(modelArch, "w") as json_file:
            json_file.write(model_json)

        'Save the weights'
        model.save_weights(os.path.join(os.path.split(path)[0], modelName, 'weights.h5'))



