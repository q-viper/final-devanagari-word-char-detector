#import libraries
import os
import numpy as np
import csv
import cv2

#define a method to wrap all essential processes, we will give location to the function
def create_csv_data_file(location):
    
    #Check if files already exists    
    exists = os.path.isfile(location[:len(location)-1] + '.csv')
    if(exists):
        print(location[:len(location)-1] +  'set already present.')
    else:
        print("Creating", location[:len(location)-1], " train file.")
        

        #iterate through the given location's every directory's list
        for each_dir in os.listdir(location):
            #again iterate through the given location's every directory's list
            for image in os.listdir(location + each_dir):

                #open every images inside that directory in grayscale
                images = location + each_dir +'/' + image 
                open_image = cv2.imread(images, 0)
                image_labels = each_dir.split('_')

                #give label to each character from 0 to 9 and ka to gya
                #digits label are from 0 to 9 while letters are 10 to 45
                if(len(image_labels) == 3): 
                    label = int(image_labels[::-1][1]) + 9
                elif(len(image_labels) == 2):
                    label = int(image_labels[::-1][0]) 

                #first column will hold the label of the example and rest 1024 will hold pixel info.
                image_array = np.hstack([np.array(label), np.array(open_image).reshape(1024)])

                #append the image information into corresponding file
                with open(location[:len(location)-1] + '.csv', 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(image_array)


create_csv_data_file('train/')
print('Done Creating trainset !!!!\n')
create_csv_data_file('test/')
print('Done Creating testset !!!!')