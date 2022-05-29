# -*- coding: utf-8 -*-

"""
FileName:               libCreationFile
Author Name:            Arun M Saranathan
Description:            This file can be used to create the spectral library files from the MICA spectra
Date Created:           19th February 2019
Last Modified:          19th February 2019
"""

import numpy as np
import spectral.io.envi as envi
from scipy import interpolate
from spectral.io.envi import SpectralLibrary as sli
import os

from generalUtilities.generalUtilities import generalUtilities
from micaTools import micaTools
from scipy.ndimage.filters import uniform_filter1d as filter1d


'Extracting CR MICA Spectrum'
sliName = '/Volume1/data/CRISM/arun/CR_MICA_Spectrum_full_v6.sli'
sliHdrName = '/Volume1/data/CRISM/arun/CR_MICA_Spectrum_full_v6.sli.hdr'

'Read in the header'
sliHdr = envi.read_envi_header(sliHdrName)
wavelength  = np.asarray(sliHdr['wavelength'], dtype='single')
wavelength  = wavelength[4:244]
wavelength  = wavelength
'Create empty variables to hold the data'
endMem_name = []
libEndMem = []
libEndMem_cont = []
libEndMem_cr = []
micaEndMem = []
micaEndMem_cont = []
micaEndMem_cr = []



'----------------------------------------------------------------------------------------------------------------------'
micaBaseFolder = '/Volume2/arunFiles/MICA Library/mrocr_8001/data/'
for r,d,f in os.walk(micaBaseFolder):
    for file in f:
        print(file)
        'Image files will have the end ".lbl"'
        if file.find('.lbl') != -1:
            lblFileName =  os.path.join(r, file)
            lblInfo = micaTools().lblRead(lblFileName)

            'extract the MICA numerator'
            fileName = lblFileName.replace('.lbl', '.tab')
            micaSpectra = micaTools().micaRead(fileName)
            f1 = interpolate.interp1d(micaSpectra[:, 0], micaSpectra[:, 5])
            micaSpectra_wvl = f1(wavelength)
            micaSpectra_wvl_smooth = filter1d(micaSpectra_wvl, 11)
            'Extract the continuum'
            points = np.vstack((wavelength, micaSpectra_wvl_smooth))
            micaSpectra_cnt = np.asarray(generalUtilities().convex_hull(points.T))


            """
            'Check and find the library spectrum source and extract data'
            libFileType = lblInfo[33][1]
            if (libFileType == '"RELAB"'):
                relabFileName = lblInfo[32][1]
                _ = relabFileName.split()
                relabFileName = _[-1]
                relabFileName = relabFileName[:-1]
                librarySpectra = micaTools().relabRead(relabFileName)
                f = interpolate.interp1d(librarySpectra[:, 0]*1, librarySpectra[:, 1])
                libSpectra_wvl = f(wavelength)

            if (libFileType == '"USGS Spectral Library"'):
                usgsFileName = lblInfo[32][1]
                # _, usgsFileName = usgsFileName.split()
                usgsFileName = usgsFileName[1:-1]
                usgsFileName = usgsFileName.lower()
                usgsFileName = usgsFileName.replace(' ', '_')
                librarySpectra = micaTools().usgsRead(usgsFileName)
                f = interpolate.interp1d(librarySpectra[:, 0]*1, librarySpectra[:, 1])
                libSpectra_wvl = f(wavelength)

            'Find the continuum'
            points = np.vstack((wavelength, libSpectra_wvl))
            libSpectra_cnt = np.asarray(generalUtilities().convex_hull(points.T))
            """

            'Place data in appropriate library'
            endMem_name.append(lblInfo[30, 1][1:-1])
            micaEndMem.append(micaSpectra_wvl)
            micaEndMem_cont.append(micaSpectra_cnt)
            micaEndMem_cr.append(micaSpectra_wvl / micaSpectra_cnt)

            """
            libEndMem.append(libSpectra_wvl)
            libEndMem_cont.append(libSpectra_cnt)
            libEndMem_cr.append((libSpectra_wvl / libSpectra_cnt))
            """

'Convert to numpy arrays'
micaEndMem = np.asarray(micaEndMem)
micaEndMem_cont = np.asarray(micaEndMem_cont)
micaEndMem_cr = np.asarray(micaEndMem_cr)
"""
libEndMem = np.asarray(libEndMem)
libEndMem_cont = np.asarray(libEndMem_cont)
libEndMem_cr = np.asarray(libEndMem_cr)
"""

'Header details'
sliHdr['wavelength'] = wavelength
sliHdr['lines'] = micaEndMem_cr.shape[0]
sliHdr['samples'] = micaEndMem_cr.shape[1]
sliHdr['spectra names'] = endMem_name

'.sli with MICA numerators'
micaNumSli = sli(micaEndMem, sliHdr, [])
micaNumSli.save('./dataProducts/micaDen')

'.sli with continuum removed MICA numerators'
micaNumSli_cr = sli(micaEndMem_cr, sliHdr, [])
micaNumSli_cr.save('./dataProducts/micaDen_CR')

'.sli with continuum for MICA numerators'
micaNumSli_cont = sli(micaEndMem_cont, sliHdr, [])
micaNumSli_cont.save('./dataProducts/micaDen_Bg')

"""
'.sli with library analogue for the MICA spectra'
libSli = sli(libEndMem, sliHdr, [])
libSli.save('./dataProducts/libSpectra')

'.sli with the continuum removed library analogue for the MICA spectra'
libSli_cr = sli(libEndMem_cr, sliHdr, [])
libSli_cr.save('./dataProducts/libSpectra_cr')

'.sli with the continuum for the library analogue for the MICA spectra'
libSli_cont = sli(libEndMem_cont, sliHdr, [])
libSli_cont.save('./dataProducts/libSpectra_Bg')
"""


