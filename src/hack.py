'''
Created on Jun 27, 2014

@author: BrianTruong
'''

import numpy as np
import cv2
import sys
#from matplotlib import pyplot as plt

debug = False

def preprocess(img):
    sz = 9
    kernel = np.ones((sz,sz),np.float32)/(sz*sz)
    #filtered = cv2.filter2D(img,-1,kernel)
    filtered = cv2.GaussianBlur(img,(sz,sz),0)
    
    descriptorExtractor = cv2.SIFT()
    
    kp, desc = descriptorExtractor.detectAndCompute(filtered, None)
    return kp, desc, filtered, cv2.drawKeypoints(filtered,kp)

def hack(imgFile1, imgFile2):
    img1 = cv2.cvtColor(cv2.imread(imgFile1, cv2.IMREAD_COLOR), cv2.COLOR_RGB2GRAY);
    img2 = cv2.cvtColor(cv2.imread(imgFile2, cv2.IMREAD_COLOR), cv2.COLOR_RGB2GRAY);
    if img1 is None or img2 is None:
        raise
    kp1, desc1, filtered1, imgProcessed1 = preprocess(img1)
    kp2, desc2, filtered2, imgProcessed2 = preprocess(img2)
    
#     if debug:
#         plt.subplot(231),plt.imshow(img1),plt.title('Original')
#         plt.xticks([]), plt.yticks([])
#         plt.subplot(232),plt.imshow(filtered1),plt.title('Filtered')
#         plt.xticks([]), plt.yticks([])
#         plt.subplot(233),plt.imshow(imgProcessed1),plt.title('Keypoints')
#         plt.xticks([]), plt.yticks([])
#         plt.subplot(234),plt.imshow(img2),plt.title('Original')
#         plt.xticks([]), plt.yticks([])
#         plt.subplot(235),plt.imshow(filtered2),plt.title('Filtered')
#         plt.xticks([]), plt.yticks([])
#         plt.subplot(236),plt.imshow(imgProcessed2),plt.title('Keypoints')
#         plt.xticks([]), plt.yticks([])
#         plt.show()
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
    
    matches = bf.match(desc1, desc2)
    return 100 * (1 - float(len(matches)) / max(len(desc1), len(desc2)))

#print hack('../distrib/set1/unchanged/pair_0000_inbound.jpg', '../distrib/set1/unchanged/pair_0000_outbound.jpg')
#print hack('../distrib/set2/unchanged/pair_0000_before.jpg', '../distrib/set2/unchanged/pair_0000_later.jpg')

arr = sys.argv
print hack(arr[0], arr[1])