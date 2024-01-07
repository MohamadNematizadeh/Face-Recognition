import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from src.FaceIdentification import FaceIdentification


app = FaceAnalysis(name="buffalo_s",providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(640,640))

face_bank_path = "./face_bank"
FaceIdentification.creat_face_bank(app,face_bank_path)             