from deepface import DeepFace
import time
import streamlit as st 
import pandas as pd
from face_function import face_reco

st.title("Facial Recognition and Verification system")

st.sidebar.image("icon.png")        
backends = [  'opencv', 
              'ssd',
              'dlib', 
              'mtcnn', 
              'retinaface', 
              'mediapipe']

models = [  "VGG-Face", 
            "Facenet",
            "Facenet512", 
            "OpenFace", 
            "DeepFace", 
            "DeepID", 
            "ArcFace", 
            "Dlib", 
            "SFace",]
metrics=st.sidebar.selectbox('witch metrics you want to use ? ',("cosine", 
                                                                "euclidean",
                                                                "euclidean_l2"))
model=st.sidebar.selectbox('witch model you want to use ? ',models)
backend=st.sidebar.selectbox('Select your backend :  ',backends)
tab0 , tab1 , tab2 = st.tabs(["Home" , "Verification" ,"Recognition"] )
with tab0 : 
	st.info("Facial Recognition and Facial Verification are two different but related technologies.")
	st.write(""" Facial Recognition is the process of identifying a person based on their face. 
	It involves comparing a live capture or digital image of a face to a database of stored images and matches it with a previously identified person.""")
	st.write(""" Facial Verification, on the other hand, is the process of confirming a user's claimed identity by comparing a live capture or digital image of their face to a stored reference image. 
	It is typically used as a means of authentication, to ensure that the person claiming to be a specific individual is indeed who they claim to be.
""")
with tab1 : 
	img_from = st.selectbox("From : " , ["image" , "camera"])
	if img_from == "camera" : 
		picture = st.camera_input("")
		if picture:
			img_name = str(time.asctime())
			with open (f'results/{img_name}.jpg','wb') as file:
				file.write(picture.getbuffer())
				try : 
					df = DeepFace.find( img_path = f"results/{img_name}.jpg",
										db_path = "DB",
										model_name = model , 
										distance_metric=metrics ,
										detector_backend =backend)
					
					probability = pd.DataFrame((1 - df[0][f"{model}_{metrics}"]) * 100)
					probability.rename(columns = {f"{model}_{metrics}":'Similarity percentage'}, inplace = True)
					id = pd.DataFrame(df[0]["identity"])
					result = pd.concat([id , probability] , axis=1).max()
					st.text(f"ID : {result[0].split('.')[0].split('/')[1]} with {round(result[1] , 2)} % probability") 
				except : 
					pass
	else : 

		impath = st.file_uploader("Epload your image file : " , ["png" , "jpg"])
		if impath:
			if st.button('start') : 
				try :
					df = DeepFace.find( img_path =impath.name ,
									db_path = "DB",
									model_name = model , 
									distance_metric=metrics , 
									detector_backend =backend)
					

					#k = df[0][df[0][f"{model}_{metrics}"] == df[0][f"{model}_{metrics}"].min() ].index[0]
					#st.write(k)
					#
					probability = pd.DataFrame((1 - df[0][f"{model}_{metrics}"]) * 100)
					probability.rename(columns = {f"{model}_{metrics}":'Similarity percentage'}, inplace = True)
					id = pd.DataFrame(df[0]["identity"])
					result = pd.concat([id , probability] , axis=1).max()
					st.text(f"ID : {result[0].split('.')[0].split('/')[1]} with {round(result[1] , 2)} % probability")          
				except :
					pass
with tab2 : 
	device = st.selectbox("Real time Frome :" , ["Camera" , "url"])
	if device == "Camera" : 
		index = st.selectbox("Index : " , (1 , 2 , 3))
	elif device == "url" : 
		index = st.text_input("Entre you Url here : ")
	if st.button("Click to start") : 
		face_reco(index=index)
