import face_recognition
import cv2
import numpy as np
import csv
import os 

video_capture = cv2.VideoCapture()

sharvari_image = face_recognition.load_image_file("sharvari.jpg")
sharvari_encoding = face_recognition.face_encodings(sharvari_image)

#known_name
known_face_encodings = [ sharvari_encoding]
known_face_names = ["sharvari"]


while True :
    _, frame = video_capture.read()
    '''small_frame = cv2.resize(frame , (0,0), fx=0.25 , fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings  = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings :
        matches = face_recognition.compare_faces(known_face_encodings , face_encodings)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encodings)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]'''

    cv2.imshow("face recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
