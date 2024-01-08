import os
import cv2
import arcade
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from src.FaceIdentification import FaceIdentification
app = FaceAnalysis(name="buffalo_s",providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(640,640))
face_bank = np.load("face_bank.npy",allow_pickle=True)
cap = cv2.VideoCapture(0)
_, frame = cap.read()
rows = frame.shape[0]
cols = frame.shape[1]
while True:
    _, input_image = cap.read()
    input_image= cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    input_image = cv2.cvtColor(input_image, cv2.COLOR_HSV2BGR)
    results = app.get(input_image)
    # print()
    FaceIdentification.face_id(results,input_image,face_bank)
    # cv2.imshow("Frame", input_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()
