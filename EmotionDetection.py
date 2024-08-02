
import cv2
from deepface import DeepFace
from base64 import b64decode

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emClass = {'neutral':'Neutral',
           'happy':'Neutral',
           'sad':'Negative',
           'angry':'Negative',
           'fear':'panic',
           'surprise':'panic'}
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
  ret, frame = cap.read()

  result = DeepFace.analyze(frame, actions=['emotion'])
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,1.1,4)
  for(x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(frame, emClass[result[0]['dominant_emotion']], (100,100), font, 3, (255, 255, 255), 2, cv2.LINE_4);
  cv2.imshow('Original video', frame)

  if cv2.waitKey(2) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()