'import spectral Python'
import spectral.io.envi as envi
import numpy as np
import os
from minMapTools import minMapTools


'------------------------------------------------------------------------------------------------------------------'
'Function to combine best guess images to form a composite best guess maps'
'------------------------------------------------------------------------------------------------------------------'
def create_Maps4CRISMImages_CompBestGuess(img55Name, img33Name=''):
    """
    @function name      :create_Maps4CRISMImages_CompBestGuess
    @description        :This function works with best guess maps with different smoothing levels. If there is a
    conflict of the mineral identified at different levels, we will revert to identification at the lowest smoothing
    levels (this will ensure that small deposits are not missed). On the other hand if the mineral identified at all
    the levels are the same the will use the highest score across the different smoothing levels.
    ----------------------------------------------------------------------------------------------------------------
    INPUTS
    ----------------------------------------------------------------------------------------------------------------
    1) img55Name:              Address of best guess image at a 5X5 level smoothing
    2) img33Name:              Address of best guess image at a 5X5 level smoothing
    3) img11Name:              Address of best guess image at a 5X5 level smoothing
    ----------------------------------------------------------------------------------------------------------------
    OUTPUTS
    ----------------------------------------------------------------------------------------------------------------
    1) outFileName:              Address of the Hyperspectral image which contains the mask image.
    """

    if not img33Name:
        img33Name = img55Name.replace('_smoothed5', '_smoothed3')

    'Read in the best guess image - 3X3 level'
    hdr33Name = img33Name.replace('.img', '.hdr')
    img_33 = envi.open(hdr33Name, img33Name)
    cube_33 = img_33.load()
    'Get size and header'
    [rows, cols, bands] = cube_33.shape
    header = envi.read_envi_header(hdr33Name)


    'Read in the best guess image - 5X5 level'
    hdr55Name = img55Name.replace('.img', '.hdr')
    img_55 = envi.open(hdr55Name, img55Name)
    cube_55 = img_55.load()

    'Get the corresponding image from the various images'
    cube_maxComposite = np.zeros((rows, cols, bands))
    cube_maxArg = np.zeros((rows, cols, bands))
    for ii in range(bands):
        temp = np.zeros((rows, cols, 3))
        temp[:, :, 0] = 0
        temp[:, :, 1] = 0
        #temp[:, :, 1] = np.squeeze(cube_33[:, :, ii])
        temp[:, :, 2] = np.squeeze(cube_55[:, :, ii])

        tempMax = np.max(temp, 2)
        tempArg = np.argmax(temp, 2)
        cube_maxComposite[:, :, ii] = tempMax
        cube_maxArg[:, :, ii] = tempArg

    'For each pixel find which mineral has highest score'
    bestGuess = np.argmax(cube_maxComposite, axis=2)
    'Now modify the cosine distance map to only hold best guess'
    for em in range(bands):
        temp = np.squeeze(cube_maxComposite[:, :, em])
        temp1 = np.squeeze(cube_maxArg[:, :, em])
        'Set all except best guess to 0'
        temp[bestGuess != em] = 0
        temp1[bestGuess != em] = 0
        cube_maxComposite[:, :, em] = temp
        cube_maxArg[:, :, em] = temp1

    'File Name for the best guess image'
    outFileName = img33Name.replace('_BestGuess', '_compBestGuess')
    outFileName = outFileName.replace('_smoothed3', '_')
    outFileName = outFileName.replace('.img', '.hdr')
    'File Name for the image which contains the smoothing level which leads to the identification'
    out1FileName = img33Name.replace('_BestGuess', '_compBestGuessArg')
    out1FileName = out1FileName.replace('_smoothed3', '_')
    out1FileName = out1FileName.replace('.img', '.hdr')

    envi.save_image(outFileName, cube_maxComposite, dtype=np.float32, force=True,
                    interleave='bil', metadata=header)
    envi.save_image(out1FileName, cube_maxArg, dtype=np.float32, force=True,
                    interleave='bil', metadata=header)

    return outFileName.replace('.hdr', '.img')

if __name__ == "__main__":
    'Set the source folder which we are to process and the target folder to '
    srcFolder =  "/Volume2/data/CRISM/AMS/v5_20200303/"

    'Set the threshold levels for the various classes'
    threshMatrix = [[0.95, 0.975], [0.8, 0.95], [0.95, 0.97], [0.83, 0.88], [0.75, 0.95], [0.75, 0.95], [0.75, 0.95],
                    [0.90, 0.97], [0.75, 0.95], [0.75, 0.95], [0.75, 0.95], [0.75, 0.95], [0.75, 0.95], [0.75, 0.95],
                    [0.75, 0.95], [0.75, 0.95], [0.75, 0.95], [0.75, 0.95], [0.95, 0.975], [0.75, 0.95], [0.9, 0.975],
                    [0.95, 0.98], [0.95, 0.98], [0.96, 0.98], [0.9, 0.975], [0.95, 0.98], [0.97, 0.99], [0.94, 0.975],
                    [0.99, 0.995], [0.99, 0.995], [0.98, 0.99], [0.98, 0.99], [0.99, 0.995], [0.99, 0.995]]
    threshMatrix = np.asarray(threshMatrix)

    'Set the colors for the various classes'
    colMat= [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0], [255, 140, 0], [255, 255, 0], [255, 0, 255],
             [255, 20, 147], [0, 0, 0], [139, 69, 19], [65, 105, 225], [0, 255, 255], [0, 255, 255], [0, 0, 0],
             [95, 158, 160], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 206, 209], [0, 0, 0], [255, 128, 0],
             [255, 128, 114], [255, 128, 114], [255, 128, 114], [0, 255, 255], [0, 0, 255], [128, 128, 0], [128, 128, 0]
             , [128, 128, 0], [128, 128, 0],  [255, 215, 0], [255, 215, 0], [255, 215, 0], [255, 215, 0]]
    colMat = np.asarray(colMat)
    imgName=''

    for r, d, f in os.walk(srcFolder):
        for file in f:
            if ((file.find('_smoothed3_CR_micaMaps_Cosine.img') != -1) or (file.find('_smoothed5_CR_micaMaps_Cosine.img') != -1)):
                'The image name is'
                imgName = os.path.join(r, file)
                'Read the image and header'
                hdrName = imgName.replace('.img', '.hdr')
                img = envi.open(hdrName, imgName)
                cube= img.load()
                header = envi.read_envi_header(hdrName)
                bands = int(header['bands'])

                'Apply appropriate threshold for each mineral'
                for ii in range(bands):
                    temp = np.squeeze(cube[:,:, ii])
                    temp[temp < threshMatrix[ii, 0]] = 0
                    cube[:, :, ii] = temp.reshape((temp.shape[0], temp.shape[1], 1))

                'Read the mask and header'
                maskName = imgName.replace('_micaMaps_Cosine', '_mask')
                maskHdr = maskName.replace('.img', '.hdr')
                mask = envi.open(maskHdr, maskName)
                mask = mask.load()
                mask = np.squeeze(mask)

                'Intitialize new products'
                cube_BestGuess = np.zeros(cube.shape)

                'For each pixel find which mineral has highest score'
                bestGuess = np.argmax(cube, axis=2)
                'Now modify the cosine distance map to only hold best guess'
                for em in range(bands):
                    temp = np.squeeze(cube[:, :, em])
                    'Set all except best guess to 0'
                    temp[bestGuess != em] = 0
                    cube_BestGuess[:, :, em] = np.multiply(temp, mask)
                    # cube_BestGuess[:,:, em] = temp

                'Save best guess cube'
                bestGuessName = imgName.replace('_Cosine', '_BestGuess_sm')
                bestGuessName = bestGuessName.replace('.img', '.hdr')
                envi.save_image(bestGuessName, cube_BestGuess, dtype=np.float32, force=True, interleave='bil',
                                metadata=header)

    for r, d, f in os.walk(srcFolder):
        for file in f:
            if file.find('_smoothed5_CR_micaMaps_BestGuess_sm.img') != -1:
                'Create Composite bestGuess Image'
                #bestGuessName = '/Volume2/data/CRISM/AMS/v5_20200105/HRL000040FF/HRL000040FF_07_IF183L_TRRD_sabcondpub_v1_lam5_cbc22_iter10002_MS_smoothed5_CR_micaMaps_BestGuess.img'
                bestGuessName = os.path.join(r, file)
                compMapName = create_Maps4CRISMImages_CompBestGuess(bestGuessName)

                compHdrName = compMapName.replace('.img', '.hdr')
                img = envi.open(compHdrName, compMapName)
                cube = img.load()
                [rows, cols, bands] = cube.shape

                'Create a colored identification maps'
                classMap = np.zeros((rows, cols, 3), dtype=np.float32)
                finalColMap = np.zeros((rows, cols, 3), dtype=np.float32)

                for ii in range(bands):
                    'Get the col map'
                    temp = colMat[ii, :]
                    band = np.squeeze(cube[:, :, ii])
                    band[band < threshMatrix[ii, 1]] = 0
                    band[band >= threshMatrix[ii, 1]] = 1

                    'Create a colored image map'
                    classMap[:, :, 0] = temp[0] * band
                    classMap[:, :, 1] = temp[1] * band
                    classMap[:, :, 2] = temp[2] * band

                    'Get the final color map'
                    finalColMap = finalColMap + classMap

                identMap = finalColMap
                classMapName = compMapName.replace('BestGuess', 'IdentMap')
                classMapName = classMapName.replace('.img', '.hdr')
                envi.save_image(classMapName, finalColMap, dtype=np.float32,
                                force=True, interleave='bil')

                'Create a colored identification maps'
                cube = img.load()
                classMap = np.zeros((rows, cols, 3), dtype=np.float32)
                finalColMap = np.zeros((rows, cols, 3), dtype=np.float32)

                for ii in range(bands):
                    'Get the col map'
                    temp = colMat[ii, :]
                    band = np.squeeze(cube[:, :, ii])
                    band[band < threshMatrix[ii, 0]]= 0
                    band[band >= threshMatrix[ii, 1]] = 0
                    band[band != 0] = 1

                    'Create a colored image map'
                    classMap[:, :, 0] = temp[0] * band
                    classMap[:, :, 1] = temp[1] * band
                    classMap[:, :, 2] = temp[2] * band

                    'Get the final color map'
                    finalColMap = finalColMap + classMap

                guessMap = finalColMap
                classMapName = compMapName.replace('BestGuess', 'GuessMap')
                classMapName = classMapName.replace('.img', '.hdr')
                envi.save_image(classMapName, finalColMap, dtype=np.float32,
                                force=True, interleave='bil')


                detectionMap = identMap + (0.6*guessMap)
                detMapName = compMapName.replace('BestGuess', 'DetectionMap')
                detMapName = detMapName.replace('.img', '.hdr')
                envi.save_image(detMapName, detectionMap, dtype=np.float32,
                                force=True, interleave='bil')


