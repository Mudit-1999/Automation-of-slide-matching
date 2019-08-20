import numpy as np
import cv2 as cv 
import os
import sys
from skimage import measure

img_mat = []
slide_mat = []
output = []
img_mat_ssim=[]
slide_mat_ssim=[]

img_match_dict = {}
slide_match_dict = {}
final_match = {}
second_match_dict= {}
i=0
j=0


s1=sys.argv[1]      # path to the directory which contains slide(soft copy)
f1=sys.argv[2]      # path to the directory which contains video frames

#  Reading all frames 

for filename in os.listdir(f1):
    if filename.endswith(".jpg"):  #specify the format of the image 
        # print(filename)
        img_match_dict[i]=filename
        img = cv.imread(f1+'/'+filename)
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
        gray= cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
        img_mat.append(gray)
        i=i+1

#  Reading all slides 

for filename in os.listdir(s1):
    if filename.endswith(".jpg"):
        print(filename)
        slide_match_dict[j]=filename
        img = cv.imread(s1+'/'+filename)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        slide_mat.append(gray)
        j=j+1

