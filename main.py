import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
# Removed unused import

path = 'student_images'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Extract class name from file name
    
    
def findEncodings(images):
    encodeList = []
    for img in images:
        encoded_faces = face_recognition.face_encodings(img)
        if encoded_faces:  # Check if encoding is not empty
            encodeList.append(encoded_faces[0])
        # Removed incorrect line
    return encodeList
encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
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
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            # since we scaled down by 4 times
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
#make a interface to show real time attendance

import tkinter as tk
from tkinter import messagebox
import pandas as pd

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System")
        self.root.geometry("300x200")
        
        self.label = tk.Label(root, text="Attendance System", font=("Helvetica", 16))
        self.label.pack(pady=20)
        
        self.show_button = tk.Button(root, text="Show Attendance", command=self.show_attendance)
        self.show_button.pack(pady=10)
        
    def show_attendance(self):
        try:
            df = pd.read_csv('Attendance.csv')
            messagebox.showinfo("Attendance", df.to_string(index=False))
        except FileNotFoundError:
            messagebox.showerror("Error", "Attendance file not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))   
            
# Create the main window

root = tk.Tk()
app = AttendanceApp(root)

root.mainloop()
cap.release()

cv2.destroyAllWindows()
