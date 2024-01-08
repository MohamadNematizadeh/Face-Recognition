import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from src.FaceIdentification import FaceIdentification

app = FaceAnalysis(name="buffalo_s",providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(640,640))

parser = argparse.ArgumentParser()
parser.add_argument('--input-image', type=str)
parser.add_argument("--update", default=False, action="store_true", help="whether perform update the dataset")
opt = parser.parse_args()


input_image = cv2.imread(opt.input_image)
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)

face_bank = np.load("face_bank.npy",allow_pickle=True)
results = app.get(input_image)
# print name_recognized_person
FaceIdentification.name_recognized_person(results,input_image,face_bank)

# Draw bounding box and name of recognized person on image
result = FaceIdentification.draw_bounding_box(results,input_image,face_bank)
face_bank = "./face_bank"
if opt.update:
        names = FaceIdentification.creat_face_bank(app,face_bank)
        print('face bank updated')
        
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
cv2.imwrite("result.jpg",result)


