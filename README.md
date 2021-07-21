# FER_webcam_4_emotes

This repository is an implementation of Facial Expression Recognition via the webcam and does the following:
  1) operate webcam using opencv library.
  2) perform facial recognition using DLib library.
  3) extract individual faces and perform FER using the Mini-Xception model (https://github.com/oarriaga/paz).
  
  For this application, the number of faces and is set to multiple but it can be tweak to one only using opencv.
  
  4) After getting the prediction from step 3, based on the softmax layer, four specific emotions such as happy, sad, angry and surprise are extracted.
  5) These emotions are then saved to a csv files along with the timestep and send them to a specified server simultaneously.
  6) The code can be ended by pressing any key and then, pressing 'q' to exit the whole app.
