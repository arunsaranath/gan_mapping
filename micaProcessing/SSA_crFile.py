# -*- coding: utf-8 -*-

"""
FileName:               SSA_crFile
Author Name:            Arun M Saranathan
Description:            This file can be used to create the CR spectral library files from the MICA SSA from Alyssa
                        Pascuzzo (Brown).
Date Created:           4th March 2019
Last Modified:          4th March 2019
"""

import numpy as np
import spectral.io.envi as envi
from spectral.io.envi import SpectralLibrary as sli
from generalUtilities.generalUtilities import generalUtilities
from micaTools import micaTools

import matplotlib.pyplot as plt

'Extracting CR MICA Spectrum'
sliName = '/Volume2/arunFiles/python_HSITools/micaProcessing/dataProducts/SSA_redMICA.sli'
sliHdrName = '/Volume2/arunFiles/python_HSITools/micaProcessing/dataProducts/SSA_redMICA.sli.hdr'

'Read in the header'
sliHdr = envi.read_envi_header(sliHdrName)
sliSSA = envi.open(sliHdrName, sliName)
spectraSSA = sliSSA.spectra
endMem_names = sliSSA.names
wavelength = np.asarray(sliHdr['wavelength'], dtype='single')
spectraSSA_CR = np.zeros(spectraSSA.shape)

'Now take each spectra look get their CR versions'
for ii in range(spectraSSA.shape[0]):
    'Extract a spectra'
    spectra = np.squeeze(spectraSSA[ii, :])

    points = np.vstack((wavelength, spectra))
    spectra_cnt = np.asarray(generalUtilities().convex_hull(points.T))
    spectraSSA_CR[ii, :] = spectra / spectra_cnt

    fig1, ax = plt.subplots(1, 2, figsize= (10, 10))
    ax[0].plot(wavelength, spectra, 'b-', linewidth=2.0)
    ax[0].plot(wavelength, spectra_cnt, 'g-.', linewidth=1.5)

    ax[1].plot(wavelength, spectra / spectra_cnt, 'r-', linewidth=2.0)
    t1 = endMem_names[ii]
    t1 = str(t1)

    fig1.savefig(('./dataProducts/SSA_ContFit/{0}.png'.format(t1)))

'.sli with MICA numerators'
micaNumSli = sli(spectraSSA_CR, sliHdr, [])
micaNumSli.save('./dataProducts/SSA_redMICA_CR')