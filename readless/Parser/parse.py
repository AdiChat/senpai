#!/usr/bin/python
# *****************************************************************************
#
# Author: Aditya Chatterjee
#
# Interweb/ contacts: GitHub.com/AdiChat
#                     Email: aditianhacker@gmail.com
#
# Helper class for reading and parsing textual data
#
# MIT License
#
# To keep up with the latest version, consult repository: GitHub.com/AdiChat/Read-Less
#
# To get an overview of this module, consult wiki: Github.com/AdiChat/Read-Less/wiki
#
# Dependencies of Parse: glob, io
#
# *****************************************************************************
import glob
import io

class Parse():

    def __init__(self):
        print "Parser for textual data"

    def dataFromFile(self, path):
        '''
        Read textual data from a file
        Arguments:
            path: path of the file (syntax is os dependent)
        Returns:
            data: a string containing data of the file
        '''
        print "Processing input " 
        text = ""
        input_files = glob.glob(path)
        for file in input_files:
            with open(file, 'r') as f:
                text = f.read()
        print "Processing completed"
        return text

    def writeDataToFile(self, pathToFile, data):
        '''
        Write textual data to file
        Arguments:
            pathToFile: path of the file to which data is to be appended
            data: the textual data to be written within the file
        Returns:
            Nothing
        Raises:
            Nothing
        '''
        File = io.open(pathToFile, 'w')
        File.write(data)
        File.close()

    def writeListDataToFile(self, pathToFile, Listdata):
        '''
        Write list of textual data to a file
        Arguments:
            pathToFile: path of the file to which data is to be appended
            Listdata: the list of textual data to be written within the file
        Returns:
            Nothing
        Raises:
            Nothing
        '''
        File = io.open(pathToFile, 'a')
        for item in Listdata:
            File.write(item)
        File.close()

    def dataFromFolder(self, path):
        '''
        Read textual data from every file within a Folder
        Arguments:
            path: path of the folder containing the files (syntax is os dependent)
        Returns:
            data: a list containing data of each file
        '''
        print "Processing input " 
        text = ""
        data = []
        counter = -1
        input_files = glob.glob(path)
        for file in input_files:
            counter += 1
            with open(file, 'r') as f:
                text = f.read()
            data.append(text)
        print "Processing completed"
        return data