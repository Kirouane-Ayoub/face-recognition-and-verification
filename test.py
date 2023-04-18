import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from deepface import DeepFace

df = DeepFace.find( img_path ="test_image.jpg",
										db_path = "DB",
										model_name = "VGG-Face" , 
										distance_metric="cosine" ,
										detector_backend ="mediapipe")
print(df[0]["identity"][0])