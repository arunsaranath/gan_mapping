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
from spectral.io.envi import SpectralLibrary as sli
import os

from micaProcessing import micaTools


'Extracting CR MICA Spectrum'
sliName = '/Volume1/data/CRISM/arun/CR_MICA_Spectrum_full_v6.sli'
sliHdrName = '/Volume1/data/CRISM/arun/CR_MICA_Spectrum_full_v6.sli.hdr'

'Read in the header'
sliHdr = envi.read_envi_header(sliHdrName)
wavelength  = np.asarray(sliHdr['wavelength'], dtype='single')
wavelength  = wavelength[4:244]
'Create empty variables to hold the data'
yukiMicaEndMem = []
endMem_name = []



'----------------------------------------------------------------------------------------------------------------------'
micaBaseFolder = '/Volume2/arunFiles/MICA Library/mrocr_8001/data/'
for r,d,f in os.walk(micaBaseFolder):
    for file in f:
        print(file)
        'Image files will have the end ".lbl"'
        if file.find('.lbl') != -1:
            lblFileName =  os.path.join(r, file)
            lblInfo = micaTools().lblRead(lblFileName)

            imgName = lblInfo[0:116][19]
            imgName = imgName[1]
            imgName = imgName[1:12]

            numColumn = lblInfo[0:116][20]
            numColumn = numColumn[1]
            numColumn = int(numColumn[:-8])

            numRow = lblInfo[0:116][21]
            numRow = numRow[1]
            numRow = int(numRow[:-8])

            roiSize = lblInfo[0:116][28]
            roiSize = roiSize[1]
            xidx = roiSize.find('x')
            roiSize = int(roiSize[1:xidx])
            roiSize = (roiSize-1)/2

            imgFolder = os.path.join('/Volume1/data/CRISM/arun/sabcondv3/trainingSet_Trail/', imgName)

            if(os.path.isdir(imgFolder)):
                for _, _, f1 in os.walk(imgFolder):
                    for file1 in f1:
                        if file1.find('_nr.img') != -1:
                            imgName = os.path.join(imgFolder, file1)
                            hdrName = imgName.replace('.img', '.hdr')

                            img = envi.open(hdrName, imgName)
                            cube = img.load()

                            roi = np.asarray(cube[(numRow-roiSize):(numRow+roiSize+1), (numColumn-roiSize):(numColumn+roiSize+1),
                                  4:244])
                            roi = roi.transpose(2, 0, 1).reshape(240, -1)

                            spectra = np.squeeze(np.mean(roi, 1))

                            endMem_name.append(lblInfo[30, 1][1:-1])
                            yukiMicaEndMem.append(spectra)



'Convert to numpy arrays'
yukiMicaEndMem = np.asarray(yukiMicaEndMem)


'Header details'
sliHdr['wavelength'] = wavelength
sliHdr['lines'] = yukiMicaEndMem.shape[0]
sliHdr['samples'] = yukiMicaEndMem.shape[1]
sliHdr['spectra names'] = endMem_name

'.sli with MICA numerators'
micaNumSli = sli(yukiMicaEndMem, sliHdr, [])
micaNumSli.save('./dataProducts/yukiMicaNum')