# File to perform real-time face mask detection using webcam feed with alert sound for no mask detection

import cv2, numpy as np
from tensorflow.keras.models import load_model # type: ignore
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from src.utils import load_face_detector, sound_alert

model = load_model("outputs/mask_detector_model.keras")
face_cascade = load_face_detector("src/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        face = cv2.resize(roi, (128,128)) / 255.0
        pred = model.predict(np.expand_dims(face,0), verbose=0)[0][0]
        label = "Mask" if pred > 0.5 else "No Mask"
        color = (0,255,0) if label=="Mask" else (0,0,255)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,f"{label}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        if label=="No Mask":
            sound_alert()

    cv2.imshow("Face Mask Detection - Live Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
