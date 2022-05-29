# -*- coding: utf-8 -*-

"""
FileName:               copyFiles_fromSource
Author Name:            Arun M Saranathan
Description:            This code file copies file from some source and places them in the target folder.

Date Created:           09th April 2019
Last Modified:          09th April 2019
"""


import os
import shutil

srcFolder = "/Volume1/data/CRISM/AMS/crism_challenge/"
trgtFolder = "/Volume2/data/CRISM/AMS/crism_challenge/"

'Look for the Modeled spectrum files'
for r,d,f in os.walk(srcFolder):
    for file in f:
        if file.find('_ca_nr.img') != -1:
            'The image name is'
            imgName = os.path.join(r, file)
            hdrName = imgName.replace('.img', '.hdr')

            'The continuum removed image is'
            crImgName = imgName.replace('_nr.img', '_MS_CR.img')
            crHdrName = crImgName.replace('.img', '.hdr')

            #bgImgName = imgName.replace('_nr.img', '_MS_Bg.img')
            #bgHdrName = bgImgName.replace('.img', '.hdr')

            'Check the folder if it exists'
            outFolder = r.replace(srcFolder, trgtFolder)
            print(outFolder)
            if not os.path.isdir(outFolder):
                os.makedirs(outFolder)

            'Copy the image file and the CR image'
            shutil.copy(imgName, outFolder)
            shutil.copy(hdrName, outFolder)
            shutil.copy(crImgName, outFolder)
            shutil.copy(crHdrName, outFolder)
            #shutil.copy(bgImgName, outFolder)
            #shutil.copy(bgHdrName, outFolder)


