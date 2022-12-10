# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:09:48 2019

@author: Quassarian Viper
"""
import cv2
from recognition import recognition
import numpy as np
import time
import matplotlib.pyplot as plt

def camera(flag):
   # choice = print("Click spacebar for photo and anything else for video.\n")
    orig = 1
    cap = cv2.VideoCapture(0)
    tr = 0.1
    br = 0.8
    lc = 0.1
    rc = 0.8
    f = 0
    
    while(flag):    
        ret, frame = cap.read()
        if ret:
            #key event
            s = cv2.waitKey(2) & 0xFF
            
            if(chr(s) == 'x'):
                f = -1
            if(chr(s) == 'z'):
                f = 1
            
            if(chr(s) == 'a'):
                tr = tr + 0.1 * f
            if(chr(s) == 'd'):
                br = br + 0.1 * f
            if (chr(s) == 's'):
                lc = lc + 0.1 * f
            if (chr(s) == 'w'):
                rc = rc + 0.1 * f
                
                
            s_x, s_y = np.shape(frame)[0] * tr, np.shape(frame)[1] * lc
            e_x, e_y = np.shape(frame)[1] * br, np.shape(frame)[0] * rc
            s_x, s_y = np.int32(s_x), np.int32(s_y)
            e_x, e_y = np.int32(e_x), np.int32(e_y)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ogray = gray[:]
            gray = gray[s_y:e_y, s_x:e_x]
            #original = frame[s_y:e_y, s_x:e_x]
            
            if (s == 32): #space to capture image and do recognition
                time1 = time.time()
                plt.imshow(frame)
                plt.show()
                recognition(gray, 'show')
                print("In %f" %(time.time()-time1), 'sec')
            if (s == 13): #enter to do realtime recognition
                orig = 0
                cv2.destroyWindow('Project DCR')
                print("Doing RT...")
                recognition(ogray, 'no')
           
            else:
                if(orig != 0):
                    show = frame[:]
                    text = "Press 'space' to take a photo and 'enter' to do realtime(slow)."
                    text1 = "Make sure the character is inside rectangle."
                    text2 = "Press a/s/d/w for change size of rectangle and z/x to increase/decrease."
                    cv2.putText(show, text1, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 100, 200))
                    cv2.putText(show, text2, (15, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 20, 255))
                    cv2.rectangle(show, (s_x, s_y), (e_x, e_y), (0, 255, 0), 2)
                    cv2.putText(show, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 0, 255), lineType=cv2.LINE_AA) 
                    cv2.imshow('Project DCR', show)
        
        else:
            print('Trying.....\n')
            continue
         
        
        if s == 27:
            break
    cap.release()
    cv2.destroyAllWindows()