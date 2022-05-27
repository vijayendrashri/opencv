from PIL import Image
import cv2 , numpy as np
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

model = load_model('hand_gesture_model.h5')

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    face = cv2.flip(frame,1)
    #define region of interest
    roi = frame[100:300,320:480]
    cv2.imshow('roi',roi)
    
    resized = cv2.resize(face,(32,32))
    new_frame=np.expand_dims(resized,axis=0)
    pred = model.predict(new_frame)
    print("pred  === ",pred)
    result = np.argmax(pred)
    print("result  ==== ",result)
    
    cv2.rectangle(frame,(320, 100), (520, 300),(0,255,0),2)
    cv2.putText(frame,str(result), (50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
    
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
