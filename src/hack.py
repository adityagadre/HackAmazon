'''
Created on Jun 27, 2014

@author: BrianTruong
'''

import numpy as np
import cv2
import sys
#from matplotlib import pyplot as plt

debug = False

def blur(img):
    #sz = 9
    #kernel = np.ones((sz,sz),np.float32)/(sz*sz)
    #filtered = cv2.filter2D(img,-1,kernel)
    #return cv2.GaussianBlur(img,(sz,sz),3.5)
    d = 14
    sigma = 150
    return cv2.bilateralFilter(img, d, sigma, sigma)

def preprocess(img):
    descriptorExtractor = cv2.SIFT()
    
    kp, desc = descriptorExtractor.detectAndCompute(img, None)
    return kp, desc, cv2.drawKeypoints(img,kp)

def preprocess2(img):
    surf = cv2.SURF(200)
    return surf.detectAndCompute(img, None)

def hack(imgFile1, imgFile2):
    img1 = cv2.cvtColor(cv2.imread(imgFile1, cv2.IMREAD_COLOR), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(cv2.imread(imgFile2, cv2.IMREAD_COLOR), cv2.COLOR_RGB2GRAY)
    #img1 = cv2.imread(imgFile1, cv2.IMREAD_GRAYSCALE)
    #img2 = cv2.imread(imgFile2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise
    kp1, desc1, processed1 = preprocess(blur(img1))
    kp2, desc2, processed2 = preprocess(blur(img2))
    #kp1, desc1 = preprocess2(blur(img1))
    #kp2, desc2 = preprocess2(blur(img2))
    
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

    if desc1 is None or desc2 is None:
        return 0

    if len(desc1) == 0 or len(desc2) == 0:
        return 0
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    
    matches = bf.match(desc1, desc2)
    matches = [m for m in matches if m.distance < 64]
    return round(100 * (1 - float(2 * len(matches)) / (len(desc1) + len(desc2))))

''' 
    # visualize the matches
    print '#matches:', len(matches)
    dist = [m.distance for m in matches]
    
    print 'distance: min: %.3f' % min(dist)
    print 'distance: mean: %.3f' % (sum(dist) / len(dist))
    print 'distance: max: %.3f' % max(dist)
    
    # threshold: half the mean
    thres_dist = (sum(dist) / len(dist)) * 0.5
    
    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance < thres_dist]
    
    print '#selected matches:', len(sel_matches)
    
    # #####################################
    # visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    
    for m in sel_matches:
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(view, (int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1])) , (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)
    
    
    cv2.imshow("view", view)
    cv2.waitKey()
'''  
    

#print hack('../distrib/set1/unchanged/pair_0000_inbound.jpg', '../distrib/set1/unchanged/pair_0000_outbound.jpg')
#print hack('../distrib/set2/unchanged/pair_0000_before.jpg', '../distrib/set2/unchanged/pair_0000_later.jpg')
#print hack('../distrib/set3/changed/pair_0006_inbound.jpg', '../distrib/set3/changed/pair_0006_outbound.jpg')
#print hack('../distrib/set3/unchanged/pair_0938_inbound.jpg', '../distrib/set3/unchanged/pair_0938_outbound.jpg')

if __name__ == '__main__':
    arr = sys.argv
    print("{0:.0f}".format(hack(arr[1], arr[2])))