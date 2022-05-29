# -*- coding: utf-8 -*-

"""
FileName:               minMapTools
Author Name:            Arun M Saranathan
Description:            This code file identifies the feature space extraction for both the MICA exemplars and test
                        spectra and finds the simlarity of test spectra to the exemplars.

Date Created:           22nd September 2018
Last Modified:          03rd September 2019
"""

'Import libraries needed'

'General Purpose python libraries'
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosineDist
import os
from scipy import ndimage

'import spectral Python'
import spectral.io.envi as envi

'Import tensorflow and keras'
from keras.layers import *


'Import user defined libraries'
from generalUtilities import generalUtilities
from hsiUtilities import hsiUtilities


class minMapTools():
    def __init__(self, strtBand=0, numBands=240, imgType='FRT', op_postFix='_micaMaps'):
        self.strtBand = strtBand
        self.numBands = numBands
        self.stopBand = self.strtBand + self.numBands
        self.op_postFix = op_postFix + '_Cosine'

        if(imgType == 'FRT') or (imgType == 'FRS') or (imgType == 'ATO'):
            self.strtRow = 0
            self.stopRow = 0
            self.strtCol = 29
            self.stopCol = -7
        else:
            if (imgType == 'HRL') or (imgType == 'HRS'):
                self.strtRow = 0
                self.stopRow = 0
                self.strtCol = 15
                self.stopCol = -4
            else:
                if (imgType == 'Unk'):
                    self.strtRow = 0
                    self.stopRow = 0
                    self.strtCol = 0
                    self.stopCol = 0

                else:
                    raise ValueError('Unknown CRISM data Type')

    '------------------------------------------------------------------------------------------------------------------'
    'Function to Generate the Similarity Maps'
    '------------------------------------------------------------------------------------------------------------------'
    def create_Maps4CRISMImages_Cosine(self, model, mica_data, imgName, endMem_Name='', scaleFlag=True,
                                           scaleLevel=0.2):
        """
        @function name      :create_Maps4CRISMImages_Cosine
        @description        :This function accepts the discriminator model, a search library-a set of spectra to find
        similarities for and the address of a CRISM image and then creates maps that illustrates the simlarities
        between the spectra in the search library and the spectra in the CRISM Image. The function assumes that both the
        search library and image have the continnum removed using the same strategy as the spectra used to train the
        model.
        ----------------------------------------------------------------------------------------------------------------
        INPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) model        : The model which recreates the representaion of the
                          GAN-discriminator
        2) mica_data    : The library which contains the string of spectra we wish to
                          find [continuum removed]. This object is expected to be
                          of size [nSamples X nBands X 1]
        3) imgName      : The address of the CRISM Image. [continuum removed]
        4) endMem_name  : Names associated with each entry in the search library
                          (has to have length = nSamples)
        5) filetype     : The file type can be 'FRT' or 'HRL'. Default value is
                          'FRT' as this has a larger frame
        ----------------------------------------------------------------------------------------------------------------
        OUTPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) mapImag      : a 3D object which the same number of rows or columns as the CRISM imagel
                          and bands equal to the number of spectra in the search library
        """

        'Get the predictions at output layer for the MICA DATA'
        mica_dataPreds_l2 = np.asarray(model.predict(mica_data))

        'Read the image and header'
        hdrName = imgName.replace('.img', '.hdr')
        img = envi.open(hdrName, imgName)
        str1 = os.getcwd()
        header = envi.read_envi_header(str1 + ('/modelHeader.hdr'))
        cubeOrig = img.load()
        [rowOrig, colOrig, _] = cubeOrig.shape

        if (self.stopCol == 0):
            self.stopCol = colOrig

        if (self.stopRow == 0):
            self.stopRow = rowOrig


        cube = cubeOrig[self.strtRow:self.stopRow, self.strtCol:self.stopCol, self.strtBand:self.stopBand]
        cube[cube >=1] = 1

        [rows, cols, bands] = cube.shape
        'Convert to array and fill nan'
        arrImg = cube.reshape((rows * cols, bands))
        arrImg = generalUtilities().fill_nan(arrImg)

        'Scale the image spectra to a specific band depth if needed'
        if scaleFlag:
            arrImg = hsiUtilities().scaleSpectra(arrImg, scaleMin=scaleLevel)

        'Get model Prediction for this image'
        arrImg = arrImg.reshape((arrImg.shape[0], arrImg.shape[1], 1))
        imgPreds = np.asarray(model.predict(arrImg))
        'Find the cosine distance between the exemplars and the data'
        dist = np.squeeze(cosineDist(mica_dataPreds_l2, imgPreds))

        'Initialize output variable'
        simMap = np.zeros((rowOrig, colOrig, mica_data.shape[0]))
        'Reshape and form the maps for each endmember'
        for em in range(mica_dataPreds_l2.shape[0]):
            distMap = np.squeeze(dist[em, :])
            distMap = distMap.reshape((rows, cols))
            simMap[self.strtRow:self.stopRow, self.strtCol:self.stopCol, em] = distMap

        'Save this map in the same folder as the input image'
        'Change the header to hold band-names of users choice'

        if (len(endMem_Name) == mica_data.shape[0]):
            header_final = header
            try:
                header_final['band names'] = endMem_Name
                header_final['samples'] = cols
                header_final['lines'] = rows
                #header_final['offset'] = header['offset']
                #header_final['file type'] = header['file type']
                #header_final['data type'] = header['data type']
                #header_final['byte order'] = header['byte order']
                header_final['bands'] = len(endMem_Name)

            except:
                pass
        else:
            header_final = header
            try:
                header_final['wavelength'] = ''
                header_final['fwhm'] = ''
                header_final['bbl'] = ''
                header_final['bands'] = mica_data.shape[0]
            except:
                pass

        'Save the MAPS'
        mapName = imgName.replace('.img', (self.op_postFix + '.hdr'))
        envi.save_image(mapName, simMap, dtype=np.float32, force=True,
                        interleave='bil', metadata=header_final)

        return mapName.replace('.hdr', '.img')

    '------------------------------------------------------------------------------------------------------------------'
    'Function to threshold the Similarity Maps to generate the Best Guess Maps'
    '------------------------------------------------------------------------------------------------------------------'
    def create_Maps4CRISMImages_BestGuess(self, imgName, threshLevel=0.707, highThreshBands=''):
        """
        @function name      :create_Maps4CRISMImages_Best Guess
        @description        :This function accepts similiarity map as input and thresholds the different mineral
        similarity maps  and thresholds out the values below a specific similarity
        ----------------------------------------------------------------------------------------------------------------
        INPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) imgName:              Address of the similarity maps for a specific set of minerals of interest.
        2) threshLevel:          The threshold below which the similarites are inconclusive.
        3) highThreshBands:      Minerals for which the simlairty must be much higher (due to rareness of mineral or
        extreme similarity to other endmembers).
        ----------------------------------------------------------------------------------------------------------------
        OUTPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) mapName:              Address of the Hyperspectral image which contains the best guess image.
        """
        'Open the similarity map of interest'
        hdrName = imgName.replace('.img', '.hdr')
        img = envi.open(hdrName, imgName)
        header = envi.read_envi_header(hdrName)
        cube = img.load()
        [rows, cols, bands] = cube.shape

        'Apply higher thresholds for bands where these higher thresholds are important'
        if highThreshBands:
            highThreshLevel = 1 - (0.5*(1 - threshLevel))
            temp = (cube[:, :, highThreshBands])
            temp[temp < highThreshLevel] = 0
            cube[:, :, highThreshBands] = temp

        'Apply the lower threshold to all other bands'
        cube[cube < threshLevel] = 0

        'Read the mask and header'
        maskName = imgName.replace(self.op_postFix, '_mask')
        maskHdr = maskName.replace('.img', '.hdr')
        mask = envi.open(maskHdr, maskName)
        mask = mask.load()
        mask = np.squeeze(mask)

        'Intitialize new products'
        cube_BestGuess = np.zeros(cube.shape)
        cube_Identification = np.zeros(cube.shape)

        'For each pixel find which mineral has highest score'
        bestGuess = np.argmax(cube, axis=2)
        'Now modify the cosine distance map to only hold best guess'
        for em in range(bands):
            temp = np.squeeze(cube[:, :, em])
            'Set all except best guess to 0'
            temp[bestGuess != em] = 0
            cube_BestGuess[:, :, em] = np.multiply(temp, mask)
            # cube_BestGuess[:,:, em] = temp

        'Save best guess cube'
        bestGuessName = imgName.replace('_Cosine', '_BestGuess')
        bestGuessName = bestGuessName.replace('.img', '.hdr')
        envi.save_image(bestGuessName, cube_BestGuess, dtype=np.float32, force=True, interleave='bil', metadata=header)

        return bestGuessName.replace('.hdr', '.img')

    '------------------------------------------------------------------------------------------------------------------'
    'Function to identify pixel spectra with significant absorptions'
    '------------------------------------------------------------------------------------------------------------------'
    def create_Mask4CRISMImages(self, imgName, sigAbsoprtionLevel=0.99):
        """
        @function name      :create_Mask4CRISMImages
        @description        :This function accepts the smoothed continuum removed image and flag spectra which have a
        comparitively large absorptions and are worthy of further analysis.
        ----------------------------------------------------------------------------------------------------------------
        INPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) imgName:              Address of the Continuum Removed smoothed image.
        2) sigAbsoprtionLevel:   The threshold below which the spectra can be considered as having absorptions of which
        can be considered significant.
        ----------------------------------------------------------------------------------------------------------------
        OUTPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) mapName:              Address of the Hyperspectral image which contains the mask image.
        """

        hdrName = imgName.replace('.img', '.hdr')
        'Read in the background image'
        img = envi.open(hdrName, imgName)
        sCRCube = img.load()
        sCRCube = sCRCube[:, :, self.strtBand:self.stopBand]

        sCRCubeMap = np.min(sCRCube, axis=2)
        temp = np.zeros(sCRCubeMap.shape)
        temp[sCRCubeMap < sigAbsoprtionLevel] = 1

        outFileName = imgName.replace('.img', '_mask.hdr')
        envi.save_image(outFileName, temp, dtype=np.float32, force=True,
                        interleave='bil')

        return outFileName.replace('.hdr', '.img')

    '------------------------------------------------------------------------------------------------------------------'
    'Function to combine best guess images to form a composite best guess maps'
    '------------------------------------------------------------------------------------------------------------------'
    def create_Maps4CRISMImages_CompBestGuess(self, img55Name, img33Name='', img11Name=''):
        """
        @function name      :create_Maps4CRISMImages_CompBestGuess
        @description        :This function works with best guess maps with different smoothing levels. If there is a
        conflict of the mineral identified at different levels, we will revert to identification at the lowest smoothing
        levels (this will ensure that small deposits are not missed). On the other hand if the mineral identified at all
        the levels are the same the will use the highest score across the different smoothing levels.
        ----------------------------------------------------------------------------------------------------------------
        INPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) img55Name:              Address of best guess image at a 5X5 level smoothing
        2) img33Name:              Address of best guess image at a 5X5 level smoothing
        3) img11Name:              Address of best guess image at a 5X5 level smoothing
        ----------------------------------------------------------------------------------------------------------------
        OUTPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) outFileName:              Address of the Hyperspectral image which contains the mask image.
        """

        'If names follow the scheme automatically extract them'
        if not img11Name:
            img11Name = img55Name.replace('_smoothed5', '_smoothed1')

        if not img33Name:
            img33Name = img55Name.replace('_smoothed5', '_smoothed3')

        'Read in the best guess image - pixel level'
        hdr11Name = img11Name.replace('.img', '.hdr')
        img_11 = envi.open(hdr11Name, img11Name)
        cube_11 = img_11.load()
        [rows, cols, bands] = cube_11.shape
        header = envi.read_envi_header(hdr11Name)

        'Read in the best guess image - 3X3 level'
        hdr33Name = img33Name.replace('.img', '.hdr')
        img_33 = envi.open(hdr33Name, img33Name)
        cube_33 = img_33.load()


        'Read in the best guess image - 5X5 level'
        hdr55Name = img55Name.replace('.img', '.hdr')
        img_55 = envi.open(hdr55Name, img55Name)
        cube_55 = img_55.load()

        'Get the corresponding image from the various images'
        cube_maxComposite = np.zeros((rows, cols, bands))
        cube_maxArg = np.zeros((rows, cols, bands))
        for ii in range(bands):
            temp = np.zeros((rows, cols, 3))
            temp[:, :, 0] = np.squeeze(cube_11[:, :, ii])
            temp[:, :, 1] = np.squeeze(cube_33[:, :, ii])
            temp[:, :, 2] = np.squeeze(cube_55[:, :, ii])

            tempMax = np.max(temp, 2)
            tempArg = np.argmax(temp, 2)
            cube_maxComposite[:, :, ii] = tempMax
            cube_maxArg[:, :, ii] = tempArg

        'For each pixel find which mineral has highest score'
        bestGuess = np.argmax(cube_maxComposite, axis=2)
        'Now modify the cosine distance map to only hold best guess'
        for em in range(bands):
            temp = np.squeeze(cube_maxComposite[:, :, em])
            temp1 = np.squeeze(cube_maxArg[:, :, em])
            'Set all except best guess to 0'
            temp[bestGuess != em] = 0
            temp1[bestGuess != em] = 0
            cube_maxComposite[:, :, em] = temp
            cube_maxArg[:, :, em] = temp1

        'File Name for the best guess image'
        outFileName = img11Name.replace('_BestGuess', '_compBestGuess')
        outFileName = outFileName.replace('_smoothed1', '_')
        outFileName = outFileName.replace('.img', '.hdr')
        'File Name for the image which contains the smoothing level which leads to the identification'
        out1FileName = img11Name.replace('_BestGuess', '_compBestGuessArg')
        out1FileName = out1FileName.replace('_smoothed1', '_')
        out1FileName = out1FileName.replace('.img', '.hdr')

        envi.save_image(outFileName, cube_maxComposite, dtype=np.float32, force=True,
                        interleave='bil', metadata=header)
        envi.save_image(out1FileName, cube_maxArg, dtype=np.float32, force=True,
                        interleave='bil', metadata=header)

        return outFileName.replace('.hdr', '.img')

    '------------------------------------------------------------------------------------------------------------------'
    'Function to create RGB identification maps a best guess image'
    '------------------------------------------------------------------------------------------------------------------'
    def create_Maps4CRISMImages_IdentMap(self, imgName, colMat, lowConfidenceMins, identLevel=0.9):
        """
        @function name      :create_Maps4CRISMImages_IdentMap
        @description        :This function works with takes a best guess image and from each mineral map identifies the
        pixels with very high score which ensures that the probability of false detection are very low. Each mineral is
        assigned a different RGB color code and the pixels with the high confidence detections are colored based on this
        similarity.
        ----------------------------------------------------------------------------------------------------------------
        INPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) imgName:              Address of best guess image.
        2) colMat:               RGB code for each of the different minerals.
        3) lowConfidenceMins:    Minerals for which we have a lower confidence and therefore a has higher threshold
        4) identLevel:           The threshold above which the pixels are high confidence detections
        ----------------------------------------------------------------------------------------------------------------
        OUTPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) outFileName:          Address of the Hyperspectral image which contains the RGB idenification Map.
        """

        hdrName = imgName.replace('.img', '.hdr')
        img = envi.open(hdrName, imgName)
        cube = img.load()
        [rows, cols, bands] = cube.shape

        'Create a colored identification maps'
        classMap = np.zeros((rows, cols, 3), dtype=np.float32)
        finalColMap = np.zeros((rows, cols, 3), dtype=np.float32)

        'The higher threshold level is'
        identLevel_high = 1 - (0.5 * (1 - identLevel))

        for ii in range(bands):
            if ii in lowConfidenceMins:
                'Get the col map'
                temp = colMat[ii, :]
                band = np.squeeze(cube[:, :, ii])
                band[band < identLevel_high] = 0
                band[band >= identLevel_high] = 1

                'Create a colored image map'
                classMap[:, :, 0] = temp[0] * band
                classMap[:, :, 1] = temp[1] * band
                classMap[:, :, 2] = temp[2] * band

                'Get the final color map'
                finalColMap = finalColMap + classMap
            else:
                temp = colMat[ii, :]
                band = np.squeeze(cube[:, :, ii])
                band[band < identLevel] = 0
                band[band >= identLevel] = 1

                'Create a colored image map'
                classMap[:, :, 0] = temp[0] * band
                classMap[:, :, 1] = temp[1] * band
                classMap[:, :, 2] = temp[2] * band

                'Get the final color map'
                finalColMap = finalColMap + classMap

        classMapName = imgName.replace('BestGuess', 'IdentMap')
        classMapName = classMapName.replace('.img', '.hdr')
        envi.save_image(classMapName, finalColMap, dtype=np.float32,
                        force=True, interleave='bil')

        return classMapName.replace('.hdr', '.img')

    '------------------------------------------------------------------------------------------------------------------'
    'Function to create RGB guess maps a best guess image'
    '------------------------------------------------------------------------------------------------------------------'
    def create_Maps4CRISMImages_GuessMap(self, imgName, colMat, lowConfidenceMins, identLevel=0.9,
                                         guessLevel=0.707):
        """
        @function name      :create_Maps4CRISMImages_GuessMap
        @description        :This function works with takes a best guess image and from each mineral map identifies the
        pixels with intermediate score which ensures that the probability of false detection is slightly higher. Each
        mineral is assigned a different RGB color code and the pixels with the high confidence detections are colored
        based on this similarity.
        ----------------------------------------------------------------------------------------------------------------
        INPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) imgName:              Address of best guess image.
        2) colMat:               RGB code for each of the different minerals.
        3) lowConfidenceMins:    Minerals for which we have a lower confidence and therefore a has higher threshold
        4) identLevel:           The threshold above which the pixels are high confidence detections
        4) guessLevel:           The threshold above which the pixels are high confidence detections
        ----------------------------------------------------------------------------------------------------------------
        OUTPUTS
        ----------------------------------------------------------------------------------------------------------------
        1) outFileName:          Address of the Hyperspectral image which contains the RGB idenification Map.
        """

        hdrName = imgName.replace('.img', '.hdr')
        img = envi.open(hdrName, imgName)
        cube = img.load()
        [rows, cols, bands] = cube.shape

        'Create a colored identification maps'
        classMap = np.zeros((rows, cols, 3), dtype=np.float32)
        finalColMap = np.zeros((rows, cols, 3), dtype=np.float32)

        'The higher threshold level is'
        identLevel_high = 1 - (0.5 * (1 - identLevel))
        guessLevel_high = identLevel

        for ii in range(bands):
            if ii in lowConfidenceMins:
                'Get the col map'
                temp = colMat[ii, :]
                band = np.squeeze(cube[:, :, ii])
                band[band < guessLevel_high] = 0
                band[band >= identLevel_high] = 0
                band[band != 0] = 1

                'Create a colored image map'
                classMap[:, :, 0] = temp[0] * band
                classMap[:, :, 1] = temp[1] * band
                classMap[:, :, 2] = temp[2] * band

                'Get the final color map'
                finalColMap = finalColMap + classMap
            else:
                temp = colMat[ii, :]
                band = np.squeeze(cube[:, :, ii])
                band[band < guessLevel] = 0
                band[band >= identLevel] = 0
                band[band != 0] = 1

                'Create a colored image map'
                classMap[:, :, 0] = temp[0] * band
                classMap[:, :, 1] = temp[1] * band
                classMap[:, :, 2] = temp[2] * band

                'Get the final color map'
                finalColMap = finalColMap + classMap

        classMapName = imgName.replace('BestGuess', 'GuessMap')
        classMapName = classMapName.replace('.img', '.hdr')
        envi.save_image(classMapName, finalColMap, dtype=np.float32,
                        force=True, interleave='bil')

        return classMapName.replace('.hdr', '.img')













