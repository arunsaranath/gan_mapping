# -*- coding: utf-8 -*-

"""
FileName:               crismProcessing
Author Name:            Arun M Saranathan
Description:            This file includes implementation of utility function for the processing of CRISM data in terms
                        of functions like estimating the continuum,the continuum removal etc..
Date Created:           05th November 2019
Last Modified:          06th November 2019
"""

import spectral.io.envi as envi
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d as filter1d
import numba as nb

from generalUtilities import generalUtilities


@nb.jit("f4[:, :](f4[:], f4[:])")
def convex_hull_jit(wvl, spectrum):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements the algorithm CONVEXHULL(P) described by  Mark de Berg, Otfried
    Cheong, Marc van Kreveld, and Mark Overmars, in Computational Geometry:
    Algorithm and Applications, pp. 6-7 in Chapter 1

    :param points: A N X 2 matrix with the wavelengths as the first column
    :return: The convex hull vector
    """
    'The starting points be the first two points'
    xcnt, y = wvl[:2], spectrum[:2]
    'Now iterate over the other points'
    for ii in range(2, spectrum.shape[0], 1):
        'check next prospective convex hull members'
        xcnt = np.append(xcnt, wvl[ii])
        y = np.append(y, spectrum[ii])
        flag = True

        while (flag == True):
            'Check if a left turn occurs at the central member'
            a1 = (y[-2] - y[-3]) / (xcnt[-2] - xcnt[-3])
            a2 = (y[-1] - y[-2]) / (xcnt[-1] - xcnt[-2])
            if (a2 > a1):
                xcnt[-2] = xcnt[-1]
                xcnt = xcnt[:-1]
                y[-2] = y[-1]
                y = y[:-1]
                flag = (xcnt.shape[0] > 2);
            else:
                flag = False

    return np.vstack((xcnt, y))


class crismProcessing_parallel():
    def __init__(self, strtBand=4, numBands=240):
        self.strtBand = strtBand
        self.numBands = numBands
        self.stopBand = self.strtBand + self.numBands

    def fnCRISM_contRem_nb(self, imgName, filter=False, filterSize=11):
        """
        This function can be used to create continuum images for CRISM images from the WUSTL pipeline

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
                    'Calculate the convex hull'
                    cHull = convex_hull_jit(wvl, spectrum)

                    f = interpolate.interp1d(np.squeeze(cHull[0, :]), np.squeeze(cHull[1, :]))
                    ycnt = f(wvl)

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
        outFileName = imgHdrName.replace('.hdr', '_CR.hdr')
        envi.save_image(outFileName, cube_cr, dtype='single',
                        force=True, interleave='bil', metadata=header)

        return outFileName.replace('.hdr', '.img')

    def fnCRspectrum(self, wvl, spectrum):
        """
        This function can be used to create a continuum removed version of the TER images

        :param wvl: the wavelength vector
        :param spectrum: the spectrum vector
        ----------------------------------------------------------------------------------------------------------------
        OUTPUT
        ----------------------------------------------------------------------------------------------------------------
        :return: cHull: The convex hull
        """

        'Calculate the convex hull'
        cHull = convex_hull_jit(wvl, spectrum)

        return cHull


if __name__ == "__main__":
    'Get an image'
    imgName = '/Volume2/data/CRISM/AMS/packageTrail/FRT00003E12/FRT00003E12_07_IF166L_TRR3_atcr_sabcondv3_1_Lib1112_1_4_5_redAb_MS.img'
    obj = crismProcessing_parallel()
    bgFileName = obj.fnCRISM_contRem_nb(imgName)



