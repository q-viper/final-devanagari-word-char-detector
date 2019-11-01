=================================================================================================================================================================
=================================================================================================================================================================
															
															Devanagari Character and Word Recognition 
																Author: Quassarian Viper

=================================================================================================================================================================
=================================================================================================================================================================
1. Requirements:
	For dependencies open project demonstration notebook.
	Further if you want to train for yourself then you need dataset also and may be 1 day of train time. I recommend google colab for train.
	
2. Time taken:
	- Model creation: 20 days(7 hours /day at least)
		- Data Collection
		- Data Manipulation
		- Data Analysis
		- Model Design
		- Model Model Creation
		- Model train/validation
		- Model Analysis
		- Model Delivery
	
	- Recognition Model: 7 days(12 hours / day at least)
		- Image Acquistion
		- Image Preprocessing
		- Image Thresholding
		- Image Manipulation
		- Text Detection
		- Text Localization
		- Text Recognition
		
We have main.py, video_test.py, recognition.py, predictor.py python files. From main.py, we call different functions and they further calls others. Here is the complete algorithm:
A. Main.py:
1. For each program run:
    1.a. Try:
        i. Ask for image directory
        ii. Open the given directory
        iii. If the directory image exists then call recognition function
        iv. Else go to except block
    2.a. Except:
        i. Call the camera funtction
        
B. Video_test.py:
Inside camera function, capture every frame of video. And for every frame of video do:
1. Define the size of square where image should align.
2. Convert every frame to Grayscale
3. Crop the frame from the aligned area
4. Define wait key for key event
    4.a. If key is 'spacebar' then call recognition(gray, 'show')
    4.b. If key is 'enter' then call recognition(gray, 'no')
    4.c. If key is 'escape' then exit the camera window
    4.d. Else do nothing and show camera running.

C. Recognition.py:
Inside recognition function:
1. Call preprocess function and recieve segments of image, template, th_img, text_color
2. Create list for labels, accuracy and also make a copy of input image
3. For each segment:
    3.a. Call detect_text function and recieve recognize image and bordered image.
    3.b. Call prediction function and receive label of segment and accuracy.
    3.c. Append segment accuracy, label
    3.d. If accruacy is more than 80:    
        3.d.i. Call dectect_image and receive the image where that is located. 
        3.d.ii. Call localize and receive image where segement is located.
4. If the average accuracy is less than 80 then:
    4.a. Call dectect_image and receive the image where that is located.
    4.b. Call localize and receive image where segement is located.
    4.c. Call prediction function and receive label of segment and accuracy.
    
5. If the input image was photo then:
        5.a. Show RGB image
         Else:
                View the frame.
6. Show prediction and accuracy of entire process.

D. predictor.py
Inside prediction function:
    1. Open the saved model and load it
    2. Prepare the labels.
    3. Convert the input image to model's input shape
    4. Do prediction 
    5. Return label and accurracy

E. preprocess.py
1. preprocess function:
    1.a. Add gaussian blur
    1.b. Threshold image to binary
    1.c. Find the background color by checking 5 pixels from left corner
    1.d. Call borders function and receive top/down and left/right borders
    1.e. Create template image by cropping image and adding dummy borders.
    1.f. Call segmentation function and receive segnents
    1.g. Return segments, template, thresholded image and text color.
    
2. borders function
    2.a. Define the checking value by 20% of number of extreme rows, define top, down.
    2.b. Make a array with all text_color value of shape of columns.
    2.c. For each rows:
        2.c.i. If any of pixels matches with previous array then increase count, the number of useful rows. Else make count zero.
        2.c.ii. If count becomes greater or equal than check then make top = row - check and break. This is our top border.
     2.d. Reverse the image and do for each rows:
         2.d.i. If any of pixels matches with previous array then increase count, the number of useful rows. Else make count zero.
         2.d.ii. If count becomes greater or equal than check then make top = row + check and break. This is our bottom border.
         2.e. Define dummy values to prevent all background loss on borders.

3. segmentation function:
    3.a. Try:
        3.a.i. Define check as 15% of rows and we remove that 15% of top rows as they must be tick, find shape and tilt the cropped image 
        3.a.ii. Define backgroound colored array
        3.a.iii. for each row on tilted image:
            - if  that row's all value is equal to colored row then append that row
        3.a.iv. Define new keys for final segmenting rows
        3.a.v. For each key in keys:
            - if the difference between consecutive keys are greater than check then add that key to new keys.
        3.a.vi. Sort new keys
        3.a.vii. For each sorted keys:
            - untilt the bordered image and take segment between each key then append it to segmented templates by tilting.
        3.a.viii. Append first and last segments
        3.a.ix. For each segment:
            - if shape of row and column of segment is less than 5% of entire row and column then remove that segment
        3.a.x. Return segments
     3.b. Except:
         - return input image

4. detect_text function:
    4.a. Convert input image into 30 * 30 and add background border around it by 1 pixel on each side and return this image.

5. localize function:
    5.a. Take template and match it on the thresholded image
    5.b. Take the points where 80% of template matches 
    5.c. Make a rectangle around those points on the original RGB image and return localized image


        
# Devanagari-Character-Recognition
This is the final project of 7th semester BSC.CSIT.
<h1 align = 'center'>Introduction</h1>
Devanagari is the national font of Nepal and is used widely throughout the India also. It contains 10 numerals(०, १, २, ३, ४, ५, ६, ७, ८, ९) and 36 consonants (क, ख, ग, घ, ङ, च, छ, ज, झ, ञ, ट, ठ, ड, ढ, ण, त, थ, द, ध,न, प,फ, ब, भ, म, य, र, ल, व, श, ष, स, ह, क्ष, त्र, ज्ञ). Some consonants are complex and made by combine some other. However, throughout this project i considered them as single character.

The required dataset is publicly available on the <a href = 'https://web.archive.org/web/20160307001701/http://cvresearchnepal.com/wordpress/dhcd/'> link.</a> 

<h1 align = 'center'> Implementation </h1>
For this project i am using python 3.6 on anaconda. More importantly i am running it on Jupyter Notebook sometimes but <strong>Google Colab</strong> is what is used most. The dependencies and libraries i used are:<br>
Python 3.6<br>
Numpy<br>
Matplotlib<br>
OpenCv<br>
Keras<br>

<h3>For better project information start with <a href = "https://github.com/q-viper/Devanagari-Character-Recognition/blob/master/Proj_demonstration.ipynb">Project Demonstration</a>.<br> For more information please visit the drive link and feel free to edit. <a href = 'https://drive.google.com/open?id=1tB4SYR4f0-narJk-OWi1LzVRNPG8WrUB'>Click here for entire project on drive link.</a> And plese help me improve it.</h3>

	
	