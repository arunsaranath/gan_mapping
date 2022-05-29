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
    srcFolder = "/Volume1/data/CRISM/atmCorr_stable/olympia_undae/"
    trgtFolder = "/Volume1/data/CRISM/atmCorr_stable/olympia_undae/"
    sLvl = 0.2
    scaleFlag = True

    '------------------------------------------------------------------------------------------------------------------'
    'STEP-1: CREATE A NEURAL NETWORK MODEL FOR THE REPRESENTATION'
    '------------------------------------------------------------------------------------------------------------------'
    obj = GANModel_1d(img_rows=240, dropout=0.0, genFilters=250, disFilters=20,
                      filterSize=11)
    dis1 = obj.disModel_CV_L6s2()
    'Get the pre-trained weights'
    dis1.load_weights(('/Volume2/arunFiles/pythonCodeFiles/CRISM_repLearning/modelsOnYukiModel_cR_wGAN' +
                       '/Models/Model-4_small/discriminator/dis_cR_75.h5'))
    disRep = obj.disModel_CV_L6s2_rep(dis1)
    disRep.summary()

    '------------------------------------------------------------------------------------------------------------------'
    'STEP-2: LOAD AND NORMALIZE MICA CONTINUUM REMOVED DATA'
    '------------------------------------------------------------------------------------------------------------------'
    sliName = '/Volume1/data/CRISM/atmCorr_stable/olympia_undae/FRT0000C2FC/manual_exemplars_CR.sli'
    sliHdrName = '/Volume1/data/CRISM/atmCorr_stable/olympia_undae/FRT0000C2FC/manual_exemplars_CR.sli.hdr'

    'Read in the spectra'
    strt_band = 1
    micaSLI = envi.open(sliHdrName, sliName)
    mica_dataRed = micaSLI.spectra
    mica_dataRed = mica_dataRed[:, strt_band:]
    'Scale the endmembers to the band depths in the image'
    if scaleFlag:
        mica_dataRed_Scale = hsiUtilities().scaleSpectra(mica_dataRed, scaleMin=sLvl)
    else:
        mica_dataRed_Scale = mica_dataRed
    mica_dataRed_Scale = mica_dataRed_Scale.reshape(mica_dataRed_Scale.shape[0],
                                        mica_dataRed_Scale.shape[1], 1)

    'Read in the header'
    sliHdr = envi.read_envi_header(sliHdrName)
    endMem_Name = sliHdr['spectra names']
    wavelength = np.asarray(sliHdr['wavelength'], dtype='single')
    wavelength = wavelength[strt_band:]


    for r, d, f in os.walk(srcFolder):
        for file in f:
            #if file.find('nr_ds.img') != -1:
            if file.find('_nr_ds.img') != -1:
                'The image name is'
                imgName = os.path.join(r, file)
                outputFolder = r.replace(srcFolder, trgtFolder) + '/'

                '------------------------------------------------------------------------------------------------------'
                'STEP-3: CREATE MODEL IMAGE'
                '------------------------------------------------------------------------------------------------------'
                #abImgName = imgName.replace('_nr_ds.img', '_AB_ds.img')
                msImgName = imgName.replace('_nr_ds.img', '_mdl_ds.img')

                '------------------------------------------------------------------------------------------------------'
                'STEP-4: CONTINUUM REMOVAL'
                '------------------------------------------------------------------------------------------------------'
                bgImgName = crismProcessing_parallel(strt_band, 240).fnCRISM_contRem_nb(msImgName)
                crImgName = crismProcessing_parallel(strt_band, 240).fnCRISM_createCRImg(imgName, bgImgName)

                '------------------------------------------------------------------------------------------------------'
                'STEP-5: FILL IN MISSING ROWS IN SOME IMAGES THAT ARE NOT A NUMBER'
                '------------------------------------------------------------------------------------------------------'
                crInterpImageName = modelFitTools(strt_band, 240, imgType=file[:3]).crism_fillNanRows(crImgName)

                'Since we are mapping at different mapping levels'
                #for kernelSize in range(1,6,2):
                kernelSize = 5
                '--------------------------------------------------------------------------------------------------'
                'STEP-6: SMOOTHING CONTINUUM REMOVED IMAGE'
                '--------------------------------------------------------------------------------------------------'
                crSmoothImageName = modelFitTools(strt_band, 240, imgType=file[:3]).crismImgSmooth(crInterpImageName,
                                                                                           kernelSize)

                '--------------------------------------------------------------------------------------------------'
                'STEP-7: GENERATE SIMILARITY MAPS BETWEEN MICA REPRESENATIONS AND DATA REPRESENTATIONS'
                '--------------------------------------------------------------------------------------------------'
                simScoreMapName = minMapTools(strt_band, 240, imgType=file[:3]).create_Maps4CRISMImages_Cosine(disRep,
                                                                                                mica_dataRed_Scale,
                                                                                     crSmoothImageName, endMem_Name,
                                                                                     scaleFlag=scaleFlag,
                                                                                     scaleLevel=sLvl)

                'First create a mask file that will hold the pixels of interest'
                maskName = minMapTools(strt_band, 240, imgType=file[:3]).create_Mask4CRISMImages(crSmoothImageName, sigAbsoprtionLevel=0.994)

                '--------------------------------------------------------------------------------------------------'
                'STEP-8: IDENTIFY SIGNIFICANT PIXEL SPECTRA & THRESHOLD SIMILARITY MAPS TO MINIMIZE FALSE POSITIVES'
                '--------------------------------------------------------------------------------------------------'
                'First create a mask file that will hold the pixels of interest'
                maskName = minMapTools(strt_band, 240, imgType=file[:3]).create_Mask4CRISMImages(crSmoothImageName
                                                                                         , sigAbsoprtionLevel=0.992)
                'Now create the best guess map'
                bestGuessName = minMapTools(strt_band, 240,
                                            imgType=file[:3]).create_Maps4CRISMImages_BestGuess(simScoreMapName)

                """'------------------------------------------------------------------------------------------------------'
                'STEP-9: COMBINE ALL THE BEST GUESS MAPS TO CREATE A COMPOSITE BEST GUESS MAP'
                '------------------------------------------------------------------------------------------------------'
                compMapName = minMapTools(strt_band, 240, imgType=file[:3]).create_Maps4CRISMImages_CompBestGuess(bestGuessName)"""

                '------------------------------------------------------------------------------------------------------'
                'STEP-10: CREATE THE RGB IDENTIFICATION & GUESS MAPS'
                '------------------------------------------------------------------------------------------------------'
                'Intialize a RGB color code for the various minerals'
                colMat = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
                colMat = np.asarray(colMat, dtype=np.float32)

                'Initialize the low confidence bands for which we are using higher thresholds'
                lowConfBands = []
                'Generate the identification map'
                identMap = minMapTools(strt_band, 240, imgType=file[:3]).create_Maps4CRISMImages_IdentMap(bestGuessName, colMat,
                                                                                                  identLevel=0.95,
                                                                                                  lowConfidenceMins=lowConfBands)
                'Generate the Guess map'
                guessMap = minMapTools(strt_band, 240, imgType=file[:3]).create_Maps4CRISMImages_GuessMap(bestGuessName, colMat,
                                                                                                  identLevel=0.95,
                                                                                                  guessLevel=0.85,
                                                                                                  lowConfidenceMins=lowConfBands)


