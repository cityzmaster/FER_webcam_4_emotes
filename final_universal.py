from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
import numpy as np
import time
import pandas as pd
import csv
import sys
import msvcrt
from datetime import datetime
from websocket import create_connection
import json


print("*************************************************************************************")
print("*********************************WELCOME!********************************************")
print("*************************************************************************************")


def send(msg,userID):
    # send message in the following format
    message = msg
    ws.send(json.dumps(message))

def recorde(df,userID):
    p = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    classifier =load_model('fer2013_mini_XCEPTION.119-0.65.hdf5', compile=False)   
    
    newTime = time.time()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        
        # Grab a single frame of video
        ret, frame = cap.read()
        predsss = []
        dti = ""

        # FPS and timestep
        curTime = time.time()
        timestep = curTime - newTime

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        if len(rects) == 0:
            msg = userID+",0,0,0,0"
            send(msg,userID)
            time.sleep(1)

        # loop over the face detections
        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            
            if x>0 and y>0:
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class

                    preds = classifier.predict(roi)[0]
                    predsss.append(preds[0])
                    predsss.append(preds[3])
                    predsss.append(preds[4])
                    predsss.append(preds[5])
          
                else:
                    predsss = np.zeros(4)
            else:
                predsss = np.zeros(4)
                msg = [userID, str(predsss[0]), str(predsss[1]), str(predsss[2]), str(predsss[3])]
                send(msg,userID,id2,ch,ty)

            # getting date and time
            dti = int(datetime.now().strftime("%m%d%Y%H%M%S"))
            # setting the information for one row to be written into csv file
            row = [{'ID':userID,'Anger': predsss[0]*100, 'Joy': predsss[1]*100, 'Sadness': predsss[2]*100,'Surprise': predsss[3]*100,'Date/time': dti, 'Timestep':timestep}]
            # add row to csv with an index
            df = df.append(row,ignore_index=True)
            # write to csv
            df.to_csv(filef)
            # send the emotions to websocket here:
            msg = userID+","+str(round(predsss[0],3))+","+str(round(predsss[1],3))+","+str(round(predsss[2],3))+","+str(round(predsss[3],3))
            send(msg,userID)
            time.sleep(1)
           
        if msvcrt.kbhit():
            if ord(msvcrt.getch()) != None:
                cap.release()
                ws.close()
                break



if __name__ == '__main__':
    # This part enables you to use this for more than 1 user.
    userID = input("Enter username/nickname/ID: ")
    while True:
        ss = input("Press s to start and q to end: ")
        ss = ss.lower()
        if ss == "s":
            # the servername would be the link you would be accessing
            ws = create_connection(servername)
            print("Press any key to stop!")
            now = datetime.now() # current date and time
            data111= now.strftime("%m_%d_%Y_%H_%M_%S")
            loca = 'data/'
            filef = loca + data111+'.csv'
            col = ['ID','Anger', 'Joy', 'Sadness', 'Surprise', 'Date/time', 'Timestep'] 
            df = pd.DataFrame(columns=col)
            recorde(df,userID)

        elif ss == "q":
            print("Thank you! The app is ending")
            sys.exit()


