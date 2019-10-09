import numpy as np
import cv2 as cv
import os

img_mat = []
slide_mat = []

img_match_dict = {}
slide_match_dict = {}
final_match = {}
i=0
j=0


def set_pixel(im,x,y,new):
    im[x,y]=new

def quantize(im):
    for y in range(0,height-1):
        for x in range(1,width-1):
            old_pixel=im[x,y]
            if old_pixel<127:
                new_pixel=0
            else:
                new_pixel=255
            set_pixel(im,x,y,new_pixel)
            quant_err=old_pixel-new_pixel
            set_pixel(im,x+1,y,im[x+1,y]+quant_err*w1)
            set_pixel(im,x-1,y+1, im[x-1,y+1] +  quant_err*w2 )
            set_pixel(im,x,y+1, im[x,y+1] +  quant_err * w3 )
            set_pixel(im,x+1,y+1, im[x+1,y+1] +  quant_err * w4 )
    return im

    
for filename in os.listdir('sample_test/frames'):
    if filename.endswith(".jpg"): 
        print(filename)
        img_match_dict[i]=filename
        img = cv.imread('sample_test/frames/'+filename)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_mat.append(gray)
        i=i+1
        width,height,z=img.shape
        w1=7/16.0
        w2=3/16.0
        w3=5/16.0
        w4=1/16.0
        blue=img[:,:,0]
        blue=quantize(blue)
        green=img[:,:,1]
        green=quantize(green)
        red=img[:,:,2]
        red=quantize(red)
        gray1= quantize(gray)   
        image = cv.merge((blue, green, red))
        cv.imshow('merged',image)
        cv.imshow('gray',gray1)
        cv.waitKey(0)




for filename in os.listdir('sample_test/slides'):
    if filename.endswith(".jpg"):
        print(filename)
        slide_match_dict[j]=filename
        img = cv.imread('sample_test/slides/'+filename)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        slide_mat.append(gray)
        j=j+1







sift = cv.xfeatures2d.SIFT_create()

for i in range(0,len(img_mat)):
    kp1, des1 = sift.detectAndCompute(img_mat[i],None)
    max_match=0
    for j in range(0,len(slide_mat)):
        kp2, des2 = sift.detectAndCompute(slide_mat[j],None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # bf = cv.BFMatcher()
        # matches = bf.knnMatch(des1,des2, k=2)
        # # img=cv.drawKeypoints(i,kp,j)
        # %cv.imwrite('sift_keypoints.jpg',img)
        # Need to draw only good matches, so create a mask
        good=[]
        # ratio test as per Lowe's paper
        for q,(m,n) in enumerate(matches):
            if m.distance < 0.45*n.distance:
                good.append([m])

        if  max_match < len(good):
            max_match = len(good)
            final_match[img_match_dict[i]]=slide_match_dict[j]
            img3 = cv.drawMatchesKnn(img_mat[i],kp1,slide_mat[j],kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(img_match_dict[i]+'match'+ slide_match_dict[j] + '.jpg',img3)
    print(img_match_dict[i],final_match[img_match_dict[i]])





