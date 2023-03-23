import face_recognition
import cv2 
import numpy as np 
import streamlit as st 

def face_reco(index) : 
    video_capture = cv2.VideoCapture(index)
    elon_image = face_recognition.load_image_file("DB/4.jpg")
    elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

    lio_image = face_recognition.load_image_file("DB/3.jpg")
    lio_face_encoding = face_recognition.face_encodings(lio_image)[0]

    known_face_encodings = [
    salah_face_encoding,
    ayoub_face_encoding ]
    known_face_names = [
    "elon musk" , 
    "lionel messi"]
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    frame_window = st.image( [] )
    while True:
        ret, frame = video_capture.read()
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                   name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
         # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
             # Scale back up face locations since the frame we detected in was scaled to 1/4 size
             top *= 4
             right *= 4
             bottom *= 4
             left *= 4
             # Draw a box around the face
             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
             # Draw a label with a name below the face
             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
             font = cv2.FONT_HERSHEY_DUPLEX
             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
        frame_window.image(frame)
    video_capture.release()




