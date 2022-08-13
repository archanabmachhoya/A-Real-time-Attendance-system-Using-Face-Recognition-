import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

path = "images"
images = []
Name = []
myList = os.listdir(path)
print(myList)
for cur_img in myList:
    current_img = cv2.imread(f'{path}/{cur_img}')
    images.append(current_img)
    Name.append(os.path.splitext(cur_img)[0])
print(Name)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print(faceEncodings(images))

encodelist = faceEncodings(images)
print("Encoding completed!!")

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDatalist = f.readlines()
        nameList = []

        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            lStr = ""
            current_time = datetime.now()
            cStr = current_time.strftime('%I:%M:%S')

            in_time = current_time.replace(hour=10, minute=0, second=0)
            print("Late minute", in_time)
            if in_time < current_time:
                late_time_hour = current_time.hour - in_time.hour
                late_time_minute = current_time.minute - in_time.minute
                print(late_time_hour,late_time_minute)
                lStr = str(late_time_hour).zfill(2) + ':' + str(0+late_time_minute).zfill(2)
            tStr = in_time.strftime('%I:%M:%S')
            dStr = in_time.strftime('%d/%m/%Y')

            f.writelines(f'\n{name},{cStr}, {tStr}, {dStr}, {lStr}')





cap = cv2.VideoCapture(0)
while True:
     ret, frame = cap.read()
     faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
     faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

     facesCurrentFrame = face_recognition.face_locations(faces)
     encodesCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame)


     for encodeFace, faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
         matches = face_recognition.compare_faces(encodelist, encodeFace)
         faceDist = face_recognition.face_distance(encodelist, encodeFace)

         matchIndex = np.argmin(faceDist)

         if matches[matchIndex]:
             name = Name[matchIndex].upper()
             print(name)
             y1,x2,y2,x1 = faceLoc
             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
             cv2.rectangle(frame, (x1, y2-35), (x2,y2),(0,255,0),cv2.FILLED)
             cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
             attendance(name)
     cv2.imshow("camara", frame)
     if cv2.waitKey(10) ==13:
         break
cap.release()
cv2.destroyAllWindows()












