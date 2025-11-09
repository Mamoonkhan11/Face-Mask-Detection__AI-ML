# Utility functions for model training, evaluation, and alerts

import matplotlib.pyplot as plt
import cv2
import time
import threading
import os
import pygame

pygame.mixer.init()
_last_alert_time = 0 

def plot_training(history, save_dir="outputs"):
    # accuracy
    plt.figure(); plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy'); plt.legend(['train','val']); plt.savefig(f"{save_dir}/training_accuracy.png"); plt.close()
    # loss
    plt.figure(); plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
    plt.title('Loss'); plt.legend(['train','val']); plt.savefig(f"{save_dir}/training_loss.png"); plt.close()

def save_model_summary(model, path="outputs/model_summary.txt"):
    with open(path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))
    print(f" Model summary saved to {path}")

def load_face_detector(xml_path="src/haarcascade_frontalface_default.xml"):
    cascade = cv2.CascadeClassifier(xml_path)
    if cascade.empty(): raise IOError("Haar cascade XML not found. Put the file at src/haarcascade_frontalface_default.xml")
    return cascade

def sound_alert():
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time >= 1.5:
        _last_alert_time = now
        threading.Thread(target=_play, daemon=True).start()

def _play():
    sound_path = os.path.join(os.path.dirname(__file__), "..", "assets", "alert.mp3")
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
