import cv2
import numpy as np
import math
import random
# Load the image
for i in range(1,18):
    res={}
    img = cv2.imread('plate'+str(i)+'.jpg',0)
    out = np.zeros(img.shape,np.uint8)
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))

    model = cv2.KNearest()
    model.train(samples,responses)

    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("BlackWhite.jpg",th3)
    img = cv2.imread('BlackWhite.jpg')
    im3=img.copy()
    height, width, depth = img.shape

    minValidHeght=height*0.4
    minValidWidth=width*0.1
    maxValidHeght=height*0.7
    maxValidWidth=width*0.4


    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # smooth the image to avoid noises
    gray = cv2.medianBlur(gray,5)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps
    thresh = cv2.dilate(thresh,None,iterations = 3)
    thresh = cv2.erode(thresh,None,iterations = 2)

    # Find the contours


    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    lst=[]
    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        found=False
        if((w>minValidWidth or h>minValidHeght) and w<h and h<maxValidHeght and w<maxValidWidth):
        	#print x,y,w,h
        	r=random.randint(0,255);
        	g=random.randint(0,255);
        	b=random.randint(0,255);
        	for l in lst:
        		if(l[0]>x and l[2]<w and l[0]+l[2]<x+w and l[1]>y and l[3]<h and l[1]+l[3]<y+h):
        			l[0]=x;
        			l[1]=y
        			l[2]=w;
        			l[3]=h;
        			found=True
        			break
        	if(found==False):
    			lst.append([x,y,w,h])
    for l in lst:
        x,y,w,h=l
        cv2.rectangle(im3,(x,y),(x+w,y+h),(0,255,0),2)
    #	cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite("temp.jpg",crop_img)
        im = cv2.imread('temp.jpg')
        im3 = im.copy()
        height, width, depth = im3.shape
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)   
        roi = thresh[0:height, 0:width]
        roismall = cv2.resize(roi,(10,10))
        roismall = roismall.reshape((1,100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
        string = chr((int((results[0][0]))))
        res[x]=string
    stringResult=''
    lstTemp=[]
    #print len(res)
    for i in range(len(res)):
        minIndex=-1;
        for r in res:
            if(r in lstTemp):
                continue   
            if(minIndex==-1):
                minIndex=r
            elif(minIndex>r):
                minIndex=r;
        #print minIndex
        stringResult+=res[minIndex]
        if(minIndex!=-1):
            lstTemp.append(minIndex)
        
    print stringResult

#cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
#cv2.imshow('im',im3)

#cv2.waitKey(0)
# Finally show the image
#cv2.imwrite("blackWhiteDetect.jpg",thresh_color)
#cv2.imshow('res',thresh_color)


#cv2.waitKey(0)
#cv2.destroyAllWindows()