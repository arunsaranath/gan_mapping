# -*- coding: utf-8 -*-

"""
FileName:               crismProcessing
Author Name:            Arun M Saranathan
Description:            This file includes implementation of utility function for the processing of CRISM data in terms
                        of functions like estimating the continuum,the continuum removal etc..
Date Created:           05th March 2019
Last Modified:          03rd September 2019
"""

import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d as filter1d
from numba import jit, prange

from generalUtilities import generalUtilities


class crismProcessing:
    def __init__(self, strtBand=4, numBands=240):
        self.strtBand = strtBand
        self.numBands = numBands
        self.stopBand = self.strtBand + self.numBands

    def fnCRISM_contRem(self, imgName, filter=False, filterSize=11):
        """
        This function can be used to create continuum images for images from the WUSTL pipeline

        ----------------------------------------------------------------------------------------------------------------
        INPUT
        ----------------------------------------------------------------------------------------------------------------
        :param imgName: The location of the image
        :param filter: Choose whether to use some spectral smoothing (default=false)
        :param filterSize: Filter size if needed (default=11)
        :param strtBand: Where does the user wish to start the continuum (default=0)
        :param numBands: The number of bands to be included (default=240)

        ----------------------------------------------------------------------------------------------------------------
        OUTPUT
        ----------------------------------------------------------------------------------------------------------------

        :return: : outFileName: The name of the file with the convex background
        """

        'The image name is'
        imgHdrName = imgName.replace(".img", ".hdr")
        imgHdrName = imgHdrName.replace(".IMG", ".HDR")
        'Now load the image'
        img = envi.open(imgHdrName, imgName)
        header = envi.read_envi_header(imgHdrName)
        cube = img.load()
        [rows, cols, bands] = img.shape

        'Get the wavelength information'
        wvl = header['wavelength']
        wvl = wvl[self.strtBand:self.stopBand]
        wvl = np.asarray(wvl, dtype='single')

        'Create a matrix to hold the background'
        cube_bg = np.empty((rows, cols, bands), dtype=np.float32)
        cube_bg[:] = np.nan

        'For each pixel find the continuum'
        for ii in tqdm(range(rows)):
            for jj in range(cols):
                'The spectrum is'
                spectrum = np.squeeze(cube[ii, jj, self.strtBand:self.stopBand])
                if filter:
                    spectrum = filter1d(spectrum, filterSize)

                'Check if it has nans'
                flag = np.isnan(spectrum).any()

                'if not nan find the continuum'
                if not flag:
                    points = np.asarray(np.vstack((wvl, spectrum)))
                    points = points.T

                    'Find the continuum'
                    ycnt = np.asarray(generalUtilities().convex_hull(points))

                    'Place this continnum in the folder'
                    cube_bg[ii, jj, self.strtBand:self.stopBand] = ycnt

        'Save the background image'
        if filter:
            outFileName = imgName.replace('.img', ('_spFilt' + str(filterSize) + '_Bg.hdr'))
            outFileName = outFileName.replace('.IMG', ('_spFilt' + str(filterSize) + '_Bg.HDR'))
        else:
            outFileName = imgName.replace('.img', '_Bg.hdr')
            outFileName = outFileName.replace('.IMG', '_Bg.HDR')

        envi.save_image(outFileName, cube_bg, dtype='single',
                        force=True, interleave='bil', metadata=header)

        return outFileName.replace('.hdr', '.img')

    def fnCRISM_createCRImg(self, imgName, bgFileName):
        """
        This function can be used to create a continuum removed version of the TER images

        :param imgName: the simple fileName
        :param bgFileName: the associated continuum file
        ----------------------------------------------------------------------------------------------------------------
        OUTPUT
        ----------------------------------------------------------------------------------------------------------------
        :return: outFileName: The name of the file with the convex background
        """

        'The image name is'
        imgHdrName = imgName.replace(".IMG", ".HDR")
        imgHdrName = imgHdrName.replace(".img", ".hdr")
        'Now load the image'
        img = envi.open(imgHdrName, imgName)
        cube = img.load()
        [rows, cols, bands] = img.shape

        'The background image is'
        bgHdrName = bgFileName.replace(".img", ".hdr")
        bgHdrName = bgHdrName.replace(".IMG", ".HDR")
        'Now load the image'
        bgImg = envi.open(bgHdrName, bgFileName)
        header = envi.read_envi_header(bgHdrName)
        bgCube = bgImg.load()

        cube_cr = np.empty((rows, cols, bands), dtype=np.float32)
        cube_cr[:] = np.nan
        cube_cr[:, :, self.strtBand:self.stopBand] = \
            (cube[:, :, self.strtBand:self.stopBand]) / (bgCube[:, :, self.strtBand:self.stopBand])
        outFileName = bgHdrName.replace('_Bg.hdr', '_CR.hdr')
        envi.save_image(outFileName, cube_cr, dtype='single',
                        force=True, interleave='bil', metadata=header)

        return outFileName.replace('.hdr', '.img')


if __name__ == "__main__":
    'Get an image'
    imgName = '/Volume2/data/CRISM/AMS/packageTrail/FRT00003E12/FRT00003E12_07_IF166L_TRR3_atcr_sabcondv3_1_Lib1112_1_4_5_redAb_MS.img'
    obj = crismProcessing()
    bgFileName = obj.fnCRISM_contRem(imgName)



