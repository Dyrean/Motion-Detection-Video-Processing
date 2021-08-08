import cv2, pandas
from datetime import datetime

status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("VideoCapture/haarcascade_frontalface_default.xml")

first_frame = None
while True:
    check, frame = video.read()
    status = 0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    faces = face_cascade.detectMultiScale(gray_frame,
        scaleFactor=1.2, minNeighbors=5)
    for x, y, w, h in faces:
        status=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)

    status_list = status_list[-2:]
    
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    cv2.imshow('Color Frame', frame)
    key = cv2.waitKey(5)
    if key == ord('q'):
        if status==1:
            times.append(datetime.now())
        break

for i in range(0, len(times), 2):
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

df.to_csv("VideoCapture/Times-Face.cvs")
video.release()
cv2.destroyAllWindows()