# Flask app to serve real-time face mask detection video feed

from flask import Flask, render_template, Response, request
import cv2, numpy as np, os, sys
from tensorflow.keras.models import load_model  # type: ignore

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.utils import load_face_detector, sound_alert

app = Flask(__name__)

model = load_model("outputs/mask_detector_model.keras")
face_cascade = load_face_detector("src/haarcascade_frontalface_default.xml")

cam = None
camera_active = False

def get_camera():
    global cam, camera_active
    if not camera_active:
        return None

    if cam is None or not cam.isOpened():
        print(" Opening webcam...")
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Failed to open webcam")
            cam = None
    return cam

def gen():
    global camera_active
    camera = get_camera()
    if camera is None:
        print("Camera inactive, generator exiting")
        return

    while camera_active:
        ok, frame = camera.read()
        if not ok:
            print("Frame read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            face = cv2.resize(roi, (128, 128)) / 255.0
            pred = model.predict(np.expand_dims(face, 0), verbose=0)[0][0]
            label = "Mask" if pred > 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if label == "No Mask":
                sound_alert()

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    print("Generator ended — camera inactive")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    global camera_active
    camera_active = True
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop", methods=["POST"])
def stop_camera():
    global cam, camera_active
    camera_active = False
    try:
        if cam and cam.isOpened():
            print("Releasing webcam...")
            cam.release()
            cv2.destroyAllWindows()
        cam = None
    except Exception as e:
        print(f" Camera release error: {e}")
    return "Camera stopped successfully"

@app.route("/exit")
def exit_page():
    """Render exit page after stopping camera."""
    return render_template("exit.html")

if __name__ == "__main__":
    app.run(debug=True)