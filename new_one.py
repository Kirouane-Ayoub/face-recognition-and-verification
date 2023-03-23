#import streamlit as st 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from deepface import DeepFace
#result = DeepFace.verify(img1_path = "DB/kirouane ayoub.jpg", img2_path = "DB/salah bouzian.jpg")
dfs = DeepFace.find(img_path = "test.png",
                     db_path = "DB" ,
                     model_name = "Facenet" , 
                     detector_backend="opencv" ,
                     distance_metric="euclidean")
print(dfs)
