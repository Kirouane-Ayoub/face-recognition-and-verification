import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from deepface import DeepFace
import time
import streamlit as st 
import pandas as pd
from face_recognition_fun import face_recognition

st.header("Facial Recognition and Verification system")      
backends = [ 
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
with st.sidebar:
	st.image("icon.png")  
	metrics=st.selectbox('witch metrics you want to use ? ',("cosine", 
															"euclidean",
															"euclidean_l2"))
	model=st.selectbox('witch model you want to use ? ',models)
	backend=st.selectbox('Select your backend :  ',backends)

tab0 , tab1 , tab2 , tab3 = st.tabs(["Home" , "Verification" , "Analysis", "Recognition"] )
with tab0 : 
	st.info("Facial Recognition and Facial Verification are two different but related technologies.")
	st.image("face_header_sd.jpg")
	st.write(""" Facial Recognition is the process of identifying a person based on their face. 
	It involves comparing a live capture or digital image of a face to a database of stored images and matches it with a previously identified person.""")
	
	st.write(""" Facial Verification, on the other hand, is the process of confirming a user's claimed identity by comparing a live capture or digital image of their face to a stored reference image. 
	It is typically used as a means of authentication, to ensure that the person claiming to be a specific individual is indeed who they claim to be.
""")
	  
with tab1 : 
	img_from = st.selectbox("From : " , ["local" , "camera"] , key="klhyujgui")
	if img_from == "camera" : 
		picture = st.camera_input("")
		if picture:
			img_name = str(time.asctime())
			img_path = f'results/{img_name}.jpg'
			with open (img_path,'wb') as file:
				file.write(picture.getbuffer())
				try : 
					df = DeepFace.find( img_path =img_path,
										db_path = "DB",
										model_name = model , 
										distance_metric=metrics ,
										detector_backend =backend)
					#probability = pd.DataFrame((1 - df[0][f"{model}_{metrics}"]) * 100)
					#probability.rename(columns = {f"{model}_{metrics}":'Similarity percentage'}, inplace = True)
					#id = pd.DataFrame(df[0]["identity"])
					#result = pd.concat([id , probability] , axis=1).max()
					#st.text(f"ID Number : {result[0].split('.')[0].split('/')[1]} with {round(result[1] , 2)} % probability")   
					#st.dataframe(df[0].iloc[0].to_frame().transpose())
					dbimg = df[0]["identity"][0]
					col1 ,col2 = st.columns(2)
					with col1 : 
						st.image(image=img_path , width=300)
					with col2 :
						st.image(image=dbimg , width=300)

				except : 
					pass
	else : 
		impath = st.file_uploader("Epload your image file : " , ["png" , "jpg" , "jpeg"])
		if impath:
			name = impath.name
			if st.button('start' , key="AZDFGHI") : 
				try :
					df = DeepFace.find( img_path=name ,
									db_path = "DB",
									model_name = model , 
									distance_metric=metrics , 
									detector_backend =backend)
					
					#probability = pd.DataFrame((1 - df[0][f"{model}_{metrics}"]) * 100)
					#probability.rename(columns = {f"{model}_{metrics}":'Similarity percentage'}, inplace = True)
					#id = pd.DataFrame(df[0]["identity"])
					#result = pd.concat([id , probability] , axis=1).max()
					#st.text(f"ID Number : {result[0].split('.')[0].split('/')[1]} with {round(result[1] , 2)} % probability")   
					st.dataframe(df[0].iloc[0].to_frame().transpose())
					dbimg = df[0]["identity"][0]
					col1 ,col2 = st.columns(2)
					with col1 : 
						st.image(image=name , width=300)
					with col2 :
						st.image(image=dbimg , width=300)
				except :
					pass


with tab2 : 
	img_from = st.selectbox("From : " , ["local" , "camera"])
	if img_from == "local" : 
		fileupload = st.file_uploader("Upload your image : " , type=["png" , "jpg" , "jpeg"])
		if fileupload : 
			name = fileupload.name
		if st.button("start" , key="ERTHNYK?") : 
			objs = DeepFace.analyze(img_path = name, 
				actions = ['age', 'gender', 'race', 'emotion'])
			st.write(objs)
	else : 
		picture = st.camera_input("")
		if picture:
			img_name = str(time.asctime())
			img_path = f'results/{img_name}.jpg'
			with open (img_path,'wb') as file:
				file.write(picture.getbuffer())
			objs = DeepFace.analyze(img_path = img_path, 
				actions = ['age', 'gender', 'race', 'emotion'] , detector_backend=backend)
			st.write(objs)



with tab3 : 
	device = st.selectbox("Real time Frome :" , ["Camera" , "url"])
	if device == "Camera" : 
		index = st.selectbox("Index : " , (0 , 1 , 2 , 3))
	elif device == "url" : 
		index = st.text_input("Entre you Url here : ")
	if st.button("Click to start" , key="jhguifrev") : 
		face_recognition(index=index)
