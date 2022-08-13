import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

imag1 = face_recognition.load_image_file(r"C:\Users\LENOVO\PycharmProjects\Face Recognition Attendance system\images\Archana.jpg")
imag1 = cv2.cvtColor(imag1, cv2.COLOR_BGR2RGB)


dimensions = imag1.shape
type = imag1.dtype
print(type)
print("dimensions", dimensions)


resize= cv2.resize(imag1, (900, 950), interpolation= cv2.INTER_LINEAR)
print("resize", resize.shape)
print(imag1)

cv2.imshow("Archana",imag1)

cv2.waitKey()
cv2.destroyAllWindows()
