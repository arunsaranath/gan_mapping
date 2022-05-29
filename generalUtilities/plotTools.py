# -*- coding: utf-8 -*-

"""
FileName:               plotUtilities
Author Name:            Arun M Saranathan
Description:            This file includes implementation of functions to plot the necessary aspects of hyperspectral
                        images spectra and RGB composites
Date Created:           25th August 2020
Last Modified:          25th August 2020
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from spectral.io import envi


class plotTools(object):
    def plotSpectra(self, spectra2Plot, wavelength=np.asarray([]), ylabel='Values', xlabel='Index',
                    pltTitle = 'Plot', saveFlag=False, saveLoc = os.getcwd(), saveName='plot.png', figsize=(8, 6),
                    fontSize=12):
        """
        This function can be used to plot the spectra present in the matrix provided to the function in a single window.

        :param spectra2Plot: This are the matrix where each column corresponds to a spectrum to be plotted
        :param wavelength: The wavelength against which the spectra are to be plotted (default: empty)
        :param ylabel: The label for the y-axis in the plot (default: 'Values')
        :param xlabel: The label for the x-axis in the plot (default: 'Index')
        :param pltTitle: The title of the Plot (default: 'Spectrum Plot')
        :param saveFlag: whether the plot is to be save or not (default: False)
        :param saveLoc: The address of the folder where the plot is to be saved (default: Current Directory)
        :param saveName: The name of the plot image (Default: 'plot.png')
        :param figsize: Size of the figure(Default: (8, 6))
        :param fontSize: Size of the fonts in the figure (Default: 12)

        :return:
        """

        'Spectrum size'
        a = spectra2Plot.shape[0]

        'Check if wavelength is provided'
        if (wavelength.shape[0] != a):
            'Warn that wavelength if provided is not correct'
            print('The wavelength if provided is not the same length as the spectra-reverting to indicies')
            xlabel = 'Index'
            wavelength = np.arange(0, a)

        'Make this plot'
        fig1 = plt.figure(figsize=figsize)
        'Set font sizes'
        plt.rc('font', size=fontSize)
        plt.rc('xtick', labelsize=fontSize)
        plt.rc('ytick', labelsize=fontSize)
        plt.plot(wavelength, spectra2Plot, linewidth=2.5)
        plt.xlabel(xlabel, fontweight='bold', size=fontSize*1.2)
        plt.ylabel(ylabel, fontweight='bold', size=fontSize*1.2)
        plt.title(pltTitle, fontweight='bold', size=fontSize*1.5)
        'Set the figure limits'
        plt.xlim((np.min(wavelength), np.max(wavelength)))


        plt.show()

        'Check whether the plot is to be saved'
        if saveFlag:
            plotName = os.path.join(saveLoc, saveName)
            fig1.savefig(plotName)




if __name__ == "__main__":
    imgName="/Volume1/data/CRISM/yuki/v5/HRL000040FF/HRL000040FF_07_IF183L_TRRD_sabcondpub_v1_lam2_cbc11_40A2_tu2_nr_ds.img"
    hdrName = imgName.replace(".img", ".hdr")

    'Open an hyperspectral image'
    img = envi.open(hdrName, imgName)
    img = img.load()

    'Get the header'
    hdr = envi.read_envi_header(hdrName)
    wvl = hdr['wavelength']
    wvl = np.asarray(wvl[4:244], dtype= np.float16)

    'Plot this spectrum'
    spec_1 = np.squeeze(img[345, 188, 4:244])
    spec_2 = np.squeeze(img[183, 151, 4:244])
    spectrum = np.vstack((spec_1, spec_2))
    _ = plotTools().plotSpectra(spectrum.T, wavelength=wvl, ylabel='CRISM I/F', xlabel='Wavelength (microns)',
    pltTitle='CRISM pixels', saveFlag=True, figsize=(8, 15))