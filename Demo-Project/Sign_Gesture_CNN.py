#!/usr/bin/env python

import numpy as np
import cv2
from tensorflow.keras.models import load_model

classifier = load_model("CV/my_sign_gesture.h5")

# Test on actual webcam
cap = cv2.VideoCapture(0)
print(" Video Capture =================================== ",cap.isOpened())
while True:
	ret,frame = cap.read()
	frame = cv2.flip(frame,1)
	#define region of interest
	roi = frame[100:400,320:620]
	cv2.imshow('roi',roi)
	roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)

	cv2.imshow('ROI scaled and Gray',roi)
	copy = frame.copy()
	cv2.rectangle(copy,(320,100),(620,400),(255,0,0),5)

	roi = roi.reshape(1,28,28,1)
	roi = roi/255
	result =  str(classifier.predict(roi,1,verbose=0)[0])
	print("result = ", result)
	result = np.argmax(model.predict(roi,1,verbose=0))
	cv2.putText(copy,str(result),(300,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
	cv2.imshow('frame',copy)

	if cv2.waitKey(33)==27:
		cv2.destroyAllWindows()
		break

cap.release()
