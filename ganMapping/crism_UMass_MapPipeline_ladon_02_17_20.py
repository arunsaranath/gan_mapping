# -*- coding: utf-8 -*-

"""
FileName:               crismImage_ProcPipeline
Author Name:            Arun M Saranathan
Description:            This code file contains the complete pipeline to enable mapping of each CRISM image

Date Created:           09th April 2019
Last Modified:          03rd September 2019
"""

'Import Libraries'
import os

'import spectral Python'
from minMapTools import minMapTools
from modelFitTools import modelFitTools
from crismProcessing import crismProcessing_parallel
from hsiUtilities import hsiUtilities
from ganTraining import GANModel_1d
from spectral.io import envi
import numpy as np

if __name__ == "__main__":

    'Set the source folder which we are to process and the target folder to '
    srcFolder = "/Volume1/data/CRISM/yuki/v3_ladon/"
    trgtFolder = "/Volume2/data/CRISM/AMS/v3_ladon/"
    sLvl = 0.1
    scaleFlag = True

    '------------------------------------------------------------------------------------------------------'
    'STEP-1: CREATE A NEURAL NETWORK MODEL FOR THE REPRESENTATION'
    '------------------------------------------------------------------------------------------------------'
    obj = GANModel_1d(img_rows=240, dropout=0.0, genFilters=250, disFilters=20,
                      filterSize=11)
    dis1 = obj.disModel_CV_L6s2()
    'Get the pre-trained weights'
    dis1.load_weights(('/Volume2/arunFiles/pythonCodeFiles/CRISM_repLearning/modelsOnYukiModel_cR_wGAN' +
                       '/Models/Model-4_small/discriminator/dis_cR_75.h5'))
    disRep = obj.disModel_CV_L6s2_rep(dis1)
    disRep.summary()

    '------------------------------------------------------------------------------------------------------'
    'STEP-2: LOAD AND NORMALIZE MICA CONTINUUM REMOVED DATA'
    '------------------------------------------------------------------------------------------------------'
    sliName = '/Volume1/data/CRISM/arun/ladonSli_2_17_2020/UMass_ladonExemplars_02_17_2020_v2.sli'
    sliHdrName = '/Volume1/data/CRISM/arun/ladonSli_2_17_2020/UMass_ladonExemplars_02_17_2020_v2.sli.hdr'

    'Read in the spectra'
    micaSLI = envi.open(sliHdrName, sliName)
    mica_dataRed = micaSLI.spectra
    mica_dataRed = mica_dataRed[:, 4:244]
    mica_dataRed = mica_dataRed.reshape(mica_dataRed.shape[0],
                                        mica_dataRed.shape[1], 1)

    'Read in the header'
    sliHdr = envi.read_envi_header(sliHdrName)
    endMem_Name = sliHdr['spectra names']
    wavelength = np.asarray(sliHdr['wavelength'], dtype='single')
    wavelength = wavelength[4:244]
    'Scale the endmembers to the band depths in the image'
    if scaleFlag:
        mica_dataRed_Scale = hsiUtilities().scaleSpectra(mica_dataRed, scaleMin=sLvl)
    else:
        mica_dataRed_Scale = mica_dataRed

    for r, d, f in os.walk(srcFolder):
        for file in f:
            if file.find('_nr_ds.img') != -1:
                'The image name is'
                imgName = os.path.join(r, file)
                outputFolder = r.replace(srcFolder, trgtFolder) + '/'

                '------------------------------------------------------------------------------------------------------'
                'STEP-3: CREATE MODEL IMAGE'
                '------------------------------------------------------------------------------------------------------'
                abImgName = imgName.replace('_nr_ds.img', '_AB_ds.img')
                msImgName = modelFitTools(4, 240).crism_CreateModeledImage(abImgName, 1, outputFolder)

                '------------------------------------------------------------------------------------------------------'
                'STEP-4: CONTINUUM REMOVAL'
                '------------------------------------------------------------------------------------------------------'
                bgImgName = crismProcessing_parallel(4, 240).fnCRISM_contRem_nb(msImgName)
                crImgName = crismProcessing_parallel(4, 240).fnCRISM_createCRImg(imgName, bgImgName)

                '------------------------------------------------------------------------------------------------------'
                'STEP-5: FILL IN MISSING ROWS IN SOME IMAGES THAT ARE NOT A NUMBER'
                '------------------------------------------------------------------------------------------------------'
                crInterpImageName = modelFitTools(4, 240).crism_fillNanRows(crImgName)

                'Since we are mapping at different mapping levels'
                for kernelSize in range(1,6,2):
                    '--------------------------------------------------------------------------------------------------'
                    'STEP-6: SMOOTHING CONTINUUM REMOVED IMAGE'
                    '--------------------------------------------------------------------------------------------------'
                    crSmoothImageName = modelFitTools(4, 240).crismImgSmooth(crInterpImageName, kernelSize)

                    '--------------------------------------------------------------------------------------------------'
                    'STEP-7: GENERATE SIMILARITY MAPS BETWEEN MICA REPRESENATIONS AND DATA REPRESENTATIONS'
                    '--------------------------------------------------------------------------------------------------'
                    simScoreMapName = minMapTools(4, 240).create_Maps4CRISMImages_Cosine(disRep, mica_dataRed_Scale,
                                                                                         crSmoothImageName, endMem_Name,
                                                                                         scaleFlag=scaleFlag,
                                                                                         scaleLevel=sLvl)

                    '--------------------------------------------------------------------------------------------------'
                    'STEP-8: IDENTIFY SIGNIFICANT PIXEL SPECTRA & THRESHOLD SIMILARITY MAPS TO MINIMIZE FALSE POSITIVES'
                    '--------------------------------------------------------------------------------------------------'
                    'First create a mask file that will hold the pixels of interest'
                    maskName = minMapTools(4, 240).create_Mask4CRISMImages(crSmoothImageName, sigAbsoprtionLevel=0.992)