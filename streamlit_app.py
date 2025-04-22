import streamlit as st
import pandas as pd
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Path to the folder containing student images
path = 'student_images'

# Load images and class names
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Extract class name from file name

# Function to find encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        encoded_faces = face_recognition.face_encodings(img)
        if encoded_faces:  # Check if encoding is not empty
            encodeList.append(encoded_faces[0])
    return encodeList

# Encode the images
encoded_face_train = findEncodings(images)

# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

# Streamlit UI
st.title("Real-Time Attendance System")
st.sidebar.title("Options")

# Option to display attendance
if st.sidebar.button("Show Attendance"):
    try:
        df = pd.read_csv('Attendance.csv')
        st.write("### Attendance Records")
        st.dataframe(df)
    except FileNotFoundError:
        st.error("Attendance file not found.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Option to start webcam for real-time attendance
if st.sidebar.button("Start Webcam"):
    st.warning("Press 'q' to stop the webcam.")
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    st.success("Webcam stopped.")

# Run the Streamlit app using the command:
# streamlit run streamlit_app.py