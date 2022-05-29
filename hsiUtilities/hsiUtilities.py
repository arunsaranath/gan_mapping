# -*- coding: utf-8 -*-

"""
FileName:               hsiUtilities
Author Name:            Arun M Saranathan
Description:            This file includes implementation of specific utility functions which are used for CRISM data
                        processing
Date Created:           19th February 2019
Last Modified:          21st May 2020
"""
import numpy as np
import spectral.io.envi as envi
from spectral.io.envi import SpectralLibrary as sli
from scipy import ndimage
import os

from generalUtilities import generalUtilities

class hsiUtilities():
    def hsiSpatialSmoothing(self, crImgName, filtersize=5):
        """
        This function can be used to perform a spatial smoothing on an Hyperspectral image and save it.

        :param crImgName: the name of the file we want to smooth
        :param filtersize: the filter size
        :return: the name of the saved smoothed image
        """
        crHdrName = crImgName.replace('.img', '.hdr')
        header = envi.read_envi_header(crHdrName)
        'Read in the background image'
        crImg = envi.open(crHdrName, crImgName)
        crCube = crImg.load()
        [rows, cols, bands] = crCube.shape
        t1 = crImgName.rfind('/')
        temp = crImgName[(t1+1):]
        if ((temp.find('FRT') != -1) or (temp.find('FRS') != -1)):
            strtCol = 29
            stopCol = -7
        else:
            if ((temp.find('HRL') != -1) or (temp.find('HRS') != -1)):
                strtCol = 15
                stopCol = -4

        crCube = crCube[:, strtCol:stopCol, 4:244]

        'Initialize matrix to nans'
        crCube_smoothed = np.empty((rows, cols, bands), dtype=float)
        crCube_smoothed[:] = np.nan

        for ii in range(240):
            bandImg = np.squeeze(crCube[:, :, ii])
            bandImg_smooth = ndimage.uniform_filter(bandImg, size=filtersize)
            crCube_smoothed[:, strtCol:stopCol, ii + 4] = bandImg_smooth

        outFileName = crImgName.replace('.img', ('_smoothed' + str(filtersize) + '.hdr'))
        envi.save_image(outFileName, crCube_smoothed, dtype=np.float32, force=True,
                        interleave='bil', metadata=header)

        return outFileName

    def scaleSpectra(self, data, scaleMin=0.02, zeroCenter=False):
        """
        This function scales every row so that it has the same minimum value.

        :param data: A numpy matrix where the rows are individual spectra.
        :param scaleMin: The difference between the maximum and minumum values (default = 0.02)
        :param zeroCenter: Each row is shifted to be centered at 0 (default = False)
        :return: scaled matrix
        """

        'If it is a 1D matrix make it a 2D matrix'
        if len(data.shape) == 1:
            data = data.reshape((1, data.shape[0]))

        assert len(data.shape) == 2

        'First subtract 1 and set everything at 1 to 0'
        data_shft = data - 1
        'divide by the minimum in each row'
        data_min = data_shft.min(axis=1)

        data_scale = np.zeros(data.shape)

        'Scale each endmember and create plots to see what it looks like'
        for ii in range(data.shape[0]):
            temp = data_shft[ii, :] / data_min[ii]
            data_scale[ii, :] = temp * -1 * scaleMin

        if zeroCenter:
            data_scale = data_scale - (1 - (scaleMin/2))

        return (data_scale + 1)

    def hsiFlip(self, imgName):
        """
        This function can be used to flip the image upside down. This is often required in the case of CRISM images as
        the data is the pds is arranged in the reverse order

        :param imgName: The address of the image to be flipped
        ----------------------------------------------------------------------------------------------------------------
        OUTPUT
        ----------------------------------------------------------------------------------------------------------------
        :return: outFileName: The name of the file with the convex background
        """

        imgHdrName = imgName.replace(".img", ".hdr")
        'Now load the image'
        img = envi.open(imgHdrName, imgName)
        header = envi.read_envi_header(imgHdrName)
        cube = img.load()
        [_, _, bands] = img.shape

        'Get the wavelength information'
        wvl = header['wavelength']
        wvl = np.asarray(wvl, dtype=np.float32)

        'Flip the image and the wavelengths'
        cube_flip = np.flip(cube, axis=2)
        wvl = np.flip(wvl, axis=0)
        header['wavelength'] = wvl

        if header['default bands']:
            defaultBands = np.asarray(header['default bands'], dtype=np.int)
            defaultBands = bands - defaultBands
            header['default bands'] = defaultBands

        'Save the flipped data'
        outFileName = imgName.replace(".img", "_flip.hdr")
        envi.save_image(outFileName, cube_flip, dtype='single', force=True, interleave='bil', metadata=header)

        return outFileName

    def hsiNan_fill(self, imgHdrName):
        """
        This function can be used to fill in the nans based on the other data in the image.

        :param imgHdrName: location of the HDR file associated with the envi image of choice
        :return:
        """
        imgName = imgHdrName.replace('.hdr', '.img')
        header = envi.read_envi_header(imgHdrName)
        'Read in the background image'
        crImg = envi.open(imgHdrName, imgName)
        crCube = crImg.load()
        [rows, cols, bands] = crCube.shape

        arrCrImg = crCube.reshape((rows * cols, bands))
        'Fill the NaNs in the columns'
        arrCrImg =  generalUtilities().fill_nan(arrCrImg)
        'Fill the NaNs in the rows'
        arrCrImgCrop = arrCrImg[:, 4:244]
        arrCrImgCrop = generalUtilities().fill_nan(arrCrImgCrop.T)
        arrCrImg[:, 4:244] = arrCrImgCrop.T
        'Reshape to image size'
        crCube_nr = arrCrImg.reshape((rows, cols, bands))

        'Save the background image'
        outFileName1 = imgName.replace('.img', '_CRnR.hdr')
        envi.save_image(outFileName1, crCube_nr, dtype='single',
                        force=True, interleave='bil', metadata=header)

        return outFileName1

    def combineSLI(self, sli1_Name='', sli1_Hdr='', sli2_Name='', sli2_Hdr='', saveFlag=True,
                   saveadd= os.getcwd(), savename='combined'):
        """
        This function can be used to combine two different SLI (Assumes the header has the same name but ends with .hdr
        instead of .sli.)

        :param sli1_Name: Name of the first SLI
        :param sli1_Hdr: Name of header associated with first SLI
        :param sli2_Name: Name of the second SLI
        :param sli2_Hdr: Name of header associated with first SLI
        :param saveFlag: Decides if the combined SLI is saved
        :param saveadd: address to be saved at (Default= cwd)
        :parame savename: name of the combined sli (Default= combined.sli)
        :return:
        """

        'Read in first Sli and header'
        if '.sli' in sli1_Name:
            sli1 = envi.open(sli1_Hdr, sli1_Name)
            sli1_spectra =sli1.spectra

            hdr1 = envi.read_envi_header(sli1_Hdr)
            sliHdr = hdr1
        else:
            raise Exception('File name does not contain the keyphrase .sli')

        'Read in first Sli and header'
        if '.sli' in sli2_Name:
            sli2 = envi.open(sli2_Hdr, sli2_Name)
            sli2_spectra = sli2.spectra

            hdr2 = envi.read_envi_header(sli2_Hdr)
        else:
            raise Exception('File name does not contain the keyphrase .sli')

        'Check if they are defined on the same size'
        if (hdr1['samples'] == hdr2['samples']):
            'Combine the two sets of spectra'
            sli_spectra = np.vstack((sli1_spectra, sli2_spectra))

            'Create the appropriate metadata'
            'Header details'
            sliHdr['lines'] = sli_spectra.shape[0]
            sliHdr['samples'] = sli_spectra.shape[1]
            sliHdr['spectra names'] = hdr1['spectra names'] + hdr2['spectra names']

            if saveFlag:
                '.sli with MICA numerators'
                micaNumSli = sli(sli_spectra, sliHdr, [])
                micaNumSli.save(os.path.join(saveadd, savename))


        return sli_spectra

    def spectralSubsetSli(self, sli_Name='', sli_Hdr='', strtBand=4, stopBand=244, saveFlag=True):
        """
        This function extracts and saves a specific spectral subset of a spectral library.

        :param sli_Name: Name of the SLI
        :param strtBand: The starting band
        :param stopBand: the final band
        :param saveFlag: are we saving the subsetted library
        :return:
        """

        'Read in first Sli and header'
        if (('.sli' in sli_Name) and ('.hdr' in sli_Hdr)):
            sli1 = envi.open(sli_Hdr, sli_Name)
            sli_spectra = sli1.spectra

            hdr = envi.read_envi_header(sli_Hdr)
            wvl = hdr['wavelength']
            sliHdr = hdr

        'Perform the subsetting'
        sli_spectra = sli_spectra[:, strtBand:stopBand]
        wvl = wvl[strtBand:stopBand]

        'Create the appropriate metadata'
        'Header details'
        sliHdr['lines'] = sli_spectra.shape[0]
        sliHdr['samples'] = sli_spectra.shape[1]
        sliHdr['wavelength'] = wvl

        if saveFlag:
            '.sli with MICA numerators'
            micaNumSli = sli(sli_spectra, sliHdr, [])
            micaNumSli.save((sli_Name.replace('.sli', '_ss')))





if __name__ == '__main__':

    obj1 = hsiUtilities()

    sli1Name = '/Volume2/arunFiles/python_HSITools/crism_simBalanced_datasets/nonExhaustiveClassification/sliFiles/mica_addedExemplars.sli'
    sli1Hdr  = '/Volume2/arunFiles/python_HSITools/crism_simBalanced_datasets/nonExhaustiveClassification/sliFiles/mica_addedExemplars.hdr'


    sli2Name = '/Volume2/arunFiles/python_HSITools/crism_simBalanced_datasets/nonExhaustiveClassification/sliFiles/added_exemplars_3e12_ss.sli'
    sli2Hdr = '/Volume2/arunFiles/python_HSITools/crism_simBalanced_datasets/nonExhaustiveClassification/sliFiles/added_exemplars_3e12_ss.hdr'

    #_ = obj1.spectralSubsetSli(sli_Name=sli2Name, sli_Hdr=sli2Hdr)

    saveLoc = '/Volume2/arunFiles/python_HSITools/crism_simBalanced_datasets/nonExhaustiveClassification/sliFiles/'
    saveName = 'mica_addedExemplars_v1'

    
    _ = obj1.combineSLI(sli1_Name=sli1Name, sli1_Hdr=sli1Hdr, sli2_Name=sli2Name, sli2_Hdr=sli2Hdr,
                      saveadd=saveLoc, savename=saveName)
