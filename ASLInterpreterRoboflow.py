#type the following into the terminal to install roboflow before running
#pip install roboflow
#pip install opencv-python

from roboflow import Roboflow
import cv2
import sys
rf = Roboflow(api_key="Y8QjyBDl8R5s1u6eBOIn")
project = rf.workspace().project("american-sign-language-letters")
model = project.version(6).model

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

try:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        predImg_dict = model.predict(frame, confidence=40, overlap=30).json()
        for prediction in predImg_dict['predictions']: #predImg_dict['predictions'] obtains the dictionary associated with the key 'predictions'; prints the prediction for however many dictionaries there are
            # print(prediction['class']) #for just the letter
            print("predicted letter is:", prediction['class'] + "; confidence level:", prediction['confidence']) #for readability

        c = cv2.waitKey(1)
        if c == 27:
            break
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
