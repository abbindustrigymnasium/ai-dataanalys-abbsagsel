import numpy as np
import cv2

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture(0)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(grey, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.DestroyAllWindows()
# img = cv2.imread('jumpy.png', 1)
# cv2.imshow('image',img)
# k = cv2.waitKey(0) & 0xFF

# if k == 27:
#     cv2.DestroyAllWindows()
# elif k == ord('s'):
#     cv2.imwrite('jumpy_copy.png',img)
#     cv2.DestroyAllWindows()