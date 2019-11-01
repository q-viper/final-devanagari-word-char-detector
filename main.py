# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:15:12 2019
@author: Quassarian Viper
"""
from recognition import recognition
import cv2
import matplotlib.pyplot as plt
from video_test import camera
import time

try: 
    test = input('Please enter the image directory with name.\n')
    test = cv2.imread(test, 0)
    plt.imshow(cv2.cvtColor(test, cv2.COLOR_GRAY2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    time1 = time.time()
    in_img = recognition(test, 'show')
    print("In %f" %(time.time()-time1), 'sec')
except:
    print("Image not found now turning to video mode.\n")
    try:
        camera(True)
    except:
        print('Something is wrong. Try with more stable, less noise and clear picture.\n')# -*- coding: utf-8 -*-

