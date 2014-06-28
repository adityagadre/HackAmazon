import cv2
import sys
import numpy as np

arr=sys.argv

#img = cv2.imread(arr[1])
#img1=cv2.imread(arr[2])

#img = cv2.imread('pair_0005_inbound.jpg')
#img1=cv2.imread('pair_0005_outbound.jpg')

im=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
im1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

(thresh, imm)=cv2.threshold(im,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(thresh, imm1)=cv2.threshold(im1,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
a=imm;
a1=imm1;
img2=a1-a
a = cv2.medianBlur(img2,7)
a = cv2.medianBlur(a,7)

(thresh, img2)=cv2.threshold(a,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)









#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#sift = cv2.SIFT()
#kp = sift.detect(gray,None)

#img=cv2.drawKeypoints(gray,kp)
cnt=cv2.countNonZero(img2)
#print cnt
if cnt>0:
	print 100
else:
	print 0
#cv2.imwrite('sift_keypoints.jpg',img2)