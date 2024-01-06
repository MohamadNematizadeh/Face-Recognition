import os
import cv2
import arcade
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from pong import Game
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
    for result in results:
        cv2.rectangle(input_image,(int(result.bbox[0]),int(result.bbox[1])),(int(result.bbox[2]),int(result.bbox[3])),(0,255,0),2)

        for person in face_bank:
            face_bank_person_embedding = person["embedding"]
            new_person_embedding = result["embedding"]

            distance = np.sqrt(np.sum((face_bank_person_embedding - new_person_embedding) **2))
            if distance < 25:
                cv2.putText(input_image,person["name"],
                (int(result.bbox[0])-50 , int(result.bbox[1])-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2,cv2.LINE_AA)
                print("ok")
                game = Game()
                arcade.run()
                break
        else:
            cv2.putText(input_image,"Unknown",
                (int(result.bbox[0])-50 , int(result.bbox[1])-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2,cv2.LINE_AA)
            print("no")
    # cv2.imshow("Frame", input_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()