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

from openMax.crism_openmax_utils import crism_openmax_utils

if __name__ == "__main__":

    'Set the source folder which we are to process and the target folder to '
    srcFolder = "/Volume1/data/CRISM/atmCorr_stable/mawrth_vallis/HRL000043EC"
    trgtFolder = "/Volume2/data/CRISM/atmCorr_stable/mawrth_vallis/HRL000043EC"
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
    """sliName = '/Volume1/data/CRISM/arun/oldSliFiles/UMass_redMICA_CR_enhanced_paper.sli'
    sliHdrName = '/Volume1/data/CRISM/arun/oldSliFiles/UMass_redMICA_CR_enhanced_paper.hdr'

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
        mica_dataRed_Scale = mica_dataRed"""

    sliName = os.path.join('/Volume1/data/CRISM/atmCorr_stable/mawrth_vallis',
                           'manual_analysis/mawrth_vallis_exemplar.sli')

    exemplarSpectra, exemplarHdr = crism_openmax_utils().prepare_sliFile(sliFileName=sliName, contRem=True,
                                                                         sclLvl=sLvl)
    endMem_Name = exemplarHdr['spectra names']
    wavelength = np.asarray(exemplarHdr['wavelength'], dtype='single')
    mica_dataRed_Scale = crism_openmax_utils().check_featExt_input(data=exemplarSpectra)

    for r, d, f in os.walk(srcFolder):
        for file in f:
            if file.find('_nr_ds.img') != -1:
                'The image name is'
                imgName = os.path.join(r, file)
                outputFolder = r.replace(srcFolder, trgtFolder) + '/'

                '------------------------------------------------------------------------------------------------------'
                'STEP-3: CREATE MODEL IMAGE'
                '------------------------------------------------------------------------------------------------------'
                msImgName = imgName.replace('_nr_ds.img', '_mdl_ds.img')

                '------------------------------------------------------------------------------------------------------'
                'STEP-4: CONTINUUM REMOVAL'
                '------------------------------------------------------------------------------------------------------'
                bgImgName = crismProcessing_parallel(1, 240).fnCRISM_contRem_nb(msImgName)
                crImgName = crismProcessing_parallel(1, 240).fnCRISM_createCRImg(imgName, bgImgName)

                '------------------------------------------------------------------------------------------------------'
                'STEP-5: FILL IN MISSING ROWS IN SOME IMAGES THAT ARE NOT A NUMBER'
                '------------------------------------------------------------------------------------------------------'
                crInterpImageName = modelFitTools(1, 240, imgType=file[:3]).crism_fillNanRows(crImgName)

                '--------------------------------------------------------------------------------------------------'
                'STEP-6: SMOOTHING CONTINUUM REMOVED IMAGE'
                '--------------------------------------------------------------------------------------------------'
                crSmoothImageName = modelFitTools(1, 240, imgType=file[:3]).crismImgSmooth(crInterpImageName, 5)

                '--------------------------------------------------------------------------------------------------'
                'STEP-7: GENERATE SIMILARITY MAPS BETWEEN MICA REPRESENATIONS AND DATA REPRESENTATIONS'
                '--------------------------------------------------------------------------------------------------'
                simScoreMapName = minMapTools(1, 240, imgType=file[:3]).create_Maps4CRISMImages_Cosine(disRep, mica_dataRed_Scale,
                                                                                         crSmoothImageName, endMem_Name,
                                                                                         scaleFlag=scaleFlag,
                                                                                         scaleLevel=sLvl)

                '--------------------------------------------------------------------------------------------------'
                'STEP-8: IDENTIFY SIGNIFICANT PIXEL SPECTRA & THRESHOLD SIMILARITY MAPS TO MINIMIZE FALSE POSITIVES'
                '--------------------------------------------------------------------------------------------------'
                'First create a mask file that will hold the pixels of interest'
                maskName = minMapTools(1, 240, imgType=file[:3]).create_Mask4CRISMImages(crSmoothImageName,
                                                                                             sigAbsoprtionLevel=0.99)
                'Now create the best guess map'
                bestGuessName = minMapTools(1, 240, imgType=file[:3]).create_Maps4CRISMImages_BestGuess(simScoreMapName)

                '------------------------------------------------------------------------------------------------------'
                'STEP-10: CREATE THE RGB IDENTIFICATION & GUESS MAPS'
                '------------------------------------------------------------------------------------------------------'
                'Intialize a RGB color code for the various minerals'
                colMat = [[255, 0, 0], [0, 255, 0], [218, 165, 32], [189, 183, 107], [25, 25, 112], [139, 69, 19],
                [0, 0, 0], [255, 20, 147], [65, 105, 225], [0, 255, 255], [0, 0, 0], [0, 128, 128],
                [255, 255, 0], [0, 0, 0], [128, 128, 128], [0, 255, 0], [250, 128, 114], [255, 165, 0],
                [255, 127, 80], [255, 80, 127], [65, 105, 225], [65, 105, 225], [0, 255, 255], [128, 0, 0],
                [0, 0, 128], [255, 165, 0], [160, 82,45], [128, 0, 0], [255, 192, 203], [255, 192, 203],
                [65, 105, 225], [255, 192, 209],[255, 20, 147], [0, 255, 255]]
                colMat = np.asarray(colMat, dtype=np.float32)

                'Initialize the low confidence bands for which we are using higher thresholds'
                lowConfBands = [28, 29, 30]
                'Generate the identification map'
                identMap = minMapTools(1, 240, imgType=file[:3]).create_Maps4CRISMImages_IdentMap(bestGuessName, colMat, identLevel=0.95,
                                                                                lowConfidenceMins=lowConfBands)
                'Generate the Guess map'
                guessMap = minMapTools(1, 240, imgType=file[:3]).create_Maps4CRISMImages_GuessMap(bestGuessName, colMat, identLevel=0.95,
                                                                                guessLevel=0.85,
                                                                                lowConfidenceMins=lowConfBands)


