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

s1=sys.argv[1]
f1=sys.argv[2]
for filename in os.listdir(f1):
    if filename.endswith(".jpg"): 
        #print(filename)
        img_match_dict[i]=filename
        img = cv.imread(f1+'/'+filename)
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
        gray= cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
        img_mat.append(gray)
        i=i+1



for filename in os.listdir(s1):
    if filename.endswith(".jpg"):
        #print(filename)
        slide_match_dict[j]=filename
        img = cv.imread(s1+'/'+filename)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        slide_mat.append(gray)
        j=j+1


sift = cv.xfeatures2d.SIFT_create()

for i in range(0,len(img_mat)):
    kp1, des1 = sift.detectAndCompute(img_mat[i],None)
    max_match=0
    max_ratio=0
    max_ind=0
    second_match=0
    second_ind=0
    second_match_ratio=0
    for j in range(0,len(slide_mat)):
        #print("hi")
        kp2, des2 = sift.detectAndCompute(slide_mat[j],None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good=[]
        for q,(m,n) in enumerate(matches):
            if m.distance < 0.467*n.distance:
                good.append([m])

        if  max_match < len(good):
                second_match=max_match
                second_match_ratio=max_ratio
                second_ind=max_ind
                if max_match!=0:
                    second_match_dict[img_match_dict[i]]=final_match[img_match_dict[i]]
                max_match = len(good)
                max_ratio= max_match/len(kp2)
                max_ind=j
                final_match[img_match_dict[i]]=slide_match_dict[j]
                # img3 = cv.drawMatchesKnn(img_mat[i],kp1,slide_mat[j],kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # cv.imwrite(img_match_dict[i]+'match'+ slide_match_dict[j] + '.jpg',img3)
        elif second_match < len(good):
                second_match=len(good)
                second_ind=j
                second_match_ratio = second_match/len(kp2)
                second_match_dict[img_match_dict[i]]=slide_match_dict[j]
    
        # if(max_match-second_match =< 30 and max_match-second_match >= 15 and max_ratio < second_match_ratio):
                # final_match[img_match_dict[i]]=slide_match_dict[second_ind]
    #print(img_match_dict[i],final_match[img_match_dict[i]], max_match,second_match,(max_ratio >second_match_ratio))
    
    slide_mat_ssim.clear()
       
    slide_mat_ssim.append(slide_mat[second_ind])
    slide_mat_ssim.append(slide_mat[max_ind])

    #if max_match - second_match >= 10:
    # print("second time")
    maxx=-1
    for j1 in range(0,len(slide_mat_ssim)):
        height = slide_mat_ssim[j1].shape[0]
        width = slide_mat_ssim[j1].shape[1]
        ss=measure.compare_ssim(cv.resize(img_mat[i] ,(width,height)), slide_mat_ssim[j1])
        if maxx<ss:
            maxx=ss
            if j1==0:
                final_match[img_match_dict[i]]=slide_match_dict[second_ind]
            else: 
                final_match[img_match_dict[i]]=slide_match_dict[max_ind]

    #print(img_match_dict[i],final_match[img_match_dict[i]])
    output.append((img_match_dict[i], final_match[img_match_dict[i]]))

output.sort(key=lambda x: x[0])
with open('output.txt', 'w') as output_file:
    for pair in output:
        output_file.write(pair[0] + ' ' + pair[1] + '\n')
