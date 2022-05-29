# -*- coding: utf-8 -*-

"""
FileName:               generalUtilities
Author Name:            Arun M Saranathan
Description:            This file includes implementation of general utility function which are used for CRISM data
                        processing
Date Created:           19th February 2019
Last Modified:          19th February 2019
"""
import numpy as np
import os
import pandas as pd


class micaTools:
    def lblRead(self, fName):
        """
        Define a function to read the label files associated with the MICA spectra
        ----------------------------------------------------------------------------------------------------------------
        :param fName: the name of the label file to read
        :return: A numpy array which contains fields with the data from the label file
        """
        'Read a lbl file to a list'
        lines = open(fName, 'r').readlines()

        'Check if the files are empty'
        if (len(lines) == 0):
            raise ValueError('This lbl file is empty')

        'Extract and split the string pairs'
        lbl = [x.split('=') for x in lines]
        fields = []
        labels = []

        'look at all the labels'
        while (len(lbl) > 0):
            temp = lbl.pop(0)
            if (len(temp) == 2):
                field, label = temp
                labels.append(label.strip())
                fields.append(field.strip())

        return (np.vstack((np.asarray(fields), np.asarray(labels)))).T


    def micaRead(self, fName):
        """
        Define a function to  read the MICA file Data
        ----------------------------------------------------------------------------------------------------------------
        :param fName: the name of the label file to read
        :return: A numpy array which contains the MICA spectral data
        """
        'Read a lbl file to a list'
        lines = open(fName, 'r').readlines()

        'Check if the files are empty'
        if (len(lines) == 0):
            raise ValueError('This .tab file is empty')

        'Extract and split the string pairs'
        spectraInfo = [x.split(',') for x in lines]
        spectraInfo = np.array(spectraInfo, dtype=np.float)

        a = np.where(spectraInfo == 65535)
        spectraInfo[a] = np.nan

        return spectraInfo


    def relabRead(self, fName, sampCatLoc = '/Volume2/arunFiles/RelabDB2017Dec31/catalogues/Spectra_Catalogue.xls'):

        """
        Get the library spectra from the relab database
        ----------------------------------------------------------------------------------------------------------------
        :param fName: the name of the label file to read
        :param sampCatLoc: The location where the RELAB Spectral Data is present
        :return: A numpy array which contains the RELAB library data
        """

        'Get the RELAB catalogue and see the spectrum'

        sampleCatalogue = pd.ExcelFile(sampCatLoc)
        sampleCatalogue = sampleCatalogue.parse(0)

        'find the row with the required information'
        chosenRow = sampleCatalogue.loc[sampleCatalogue['SpectrumID'] == fName]

        'Get the sample information'
        sampleInfo = ((chosenRow['SampleID']).iloc[0]).lower()
        sampleInfo = sampleInfo.split('-')

        'Get the file'
        baseStr = '/Volume2/arunFiles/RelabDB2017Dec31/data/'
        addStr = baseStr + str(sampleInfo[1]) + '/' + str(sampleInfo[0] + '/')
        addStr = addStr + fName.lower() + '.txt'

        'Read a RELAB file to a list'
        lines = open(addStr, 'r').readlines()[2::]

        'Extract and split the string pairs'
        spectraInfo = [x.split() for x in lines]
        spectraInfo = np.array(spectraInfo, dtype=np.float)

        return spectraInfo

    def usgsRead(self, fName, loc='/Volume2/arunFiles/MICA Library/ASCII/M/'):
        """

        :param fName: the name of the label file to read
        :param usgsLoc: the location where the USGS folder
        :return: A numpy array which contains the USGS library data
        """
        usgsLoc = os.listdir(loc)
        sampleList = [myStr for myStr in usgsLoc if fName in myStr]

        addStr = '/Volume2/arunFiles/MICA Library/ASCII/M/' + str(sampleList[-1])

        'Read a RELAB file to a list'
        lines = open(addStr, 'r').readlines()[17::]

        'Extract and split the string pairs'
        spectraInfo = [x.split() for x in lines]
        spectraInfo = np.array(spectraInfo, dtype=np.float)

        a = np.where(spectraInfo == -12300000000000000425850770517131264.0)
        spectraInfo[a] = np.nan

        return spectraInfo  
    