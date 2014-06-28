'''
Created on Jun 27, 2014

@author: BrianTruong
'''

nsets = 3

import numpy as np
import hack
import os
import math
import random

def hackRandom(imgFile1, imgFile2):
    return 100 * random.random()

def getFolderPath(setNo, isChanged):
    return '../distrib/set{0}/{1}/'.format(setNo, 'changed' if isChanged is True else 'unchanged')

def getPath(setNo, isChanged, imgNo, isInbound):
    return getFolderPath(setNo, isChanged) + 'pair_{0:04}_{1}.jpg'.format(imgNo, 'inbound' if isInbound is True else 'outbound')

if __name__ == '__main__':
    for i in xrange(1,4):
        for isChanged in [True, False]:
            positiveCounts = np.zeros(101, dtype=int)
            
            folderPath = getFolderPath(i, isChanged)
            absFiles = [folderPath + f for f in sorted(os.listdir(folderPath))]
            imgFiles = [f for f in absFiles if os.path.isfile(f) and os.path.splitext(f)[-1].lower() == '.jpg']
            
            it = iter(imgFiles)
            imgFilePairs = zip(it, it)
            
            scores = [hack.hack(imgFilePair[0], imgFilePair[1]) for imgFilePair in imgFilePairs]
            print 'avg = ' + str(sum(scores) / len(scores))
            print scores
            
            for score in scores: 
                positiveCounts[range(int(math.ceil(100-score)), 101)] += 1;
            print positiveCounts
            
            if isChanged:
                truePositiveRates = np.divide(positiveCounts, float(len(imgFilePairs)))
                #print truePositiveRates
            else:
                falsePositiveRates = np.divide(positiveCounts, float(len(imgFilePairs)))
                #print falsePositiveRates
        print np.trapz(truePositiveRates, falsePositiveRates)
    
'''

'''