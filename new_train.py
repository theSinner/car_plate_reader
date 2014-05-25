import cv2
import numpy as np
from datetime import datetime
import os


###########################################################################
    
#prepare data for training
rps=['4','6','6','X','R','H','E','K','1','A','R','I','B','8','6',
            '5','6','1','7','W','K','B','7','7','7','P','K','A','A','7',
            '1','2','5','Q','P','F','8','4','6','0','X','W','4','1','2',
            '5','5','B','A','7','8','4','A','A','8','2','0','B','A','E',
            'B','Y','7','0','3','7','M','4','7','3','R','Q','H','9','0',
            '2','7','E','I','X','S','1','D','D','3','3','1','4','5','6',
            'R','G','6','5','R','A','0','1','J','P','7','5','A','1','H',
            'G','4','2','3']
rps=[52, 54, 54, 88, 82, 72, 69, 75, 49, 65, 82, 73, 66, 56, 54, 53, 54, 49,
    55, 87, 75, 66, 55, 55, 55, 80, 75, 65, 65, 55, 49, 50, 53, 81, 80, 70, 56,
    52, 54, 48, 88, 87, 52, 49, 50, 53, 53, 66, 65, 55, 56, 52, 65, 65, 56, 50,
    48, 66, 65, 69, 66, 89, 55, 48, 51, 55, 77, 52, 55, 51, 82, 81, 72, 57, 48,
    50, 55, 69, 73, 88, 83, 49, 68, 68, 51, 51, 49, 52, 53, 54, 82, 71, 54, 53,
    82, 65, 48, 49, 74, 80, 55, 53, 65, 49, 72, 71, 52, 50, 51]
responses = []
print len(rps)
def prepare_data():
    global responses
    fileNumber=len(rps)
    counter=0;
    samples=np.empty((0,100))
    for f in range(fileNumber):
        im = cv2.imread('./numbers/'+str(f)+'.jpg')
        im3 = im.copy()
        height, width, depth = im3.shape
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)   
        roi = thresh[0:height, 0:width]
        roismall = cv2.resize(roi,(10,10))
        responses.append(rps[f])
        sample = roismall.reshape((1,100)) 
        samples = np.append(samples,sample,0)
    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "training complete"

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
    
##########################################################
def train():
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((1,responses.size))
    
    model = cv2.KNearest()
    model.train(samples,responses)
    print ' training complete'
    return model


prepare_data()
#train()
#image_processing()
