"""
Real-time mask detection using webcam and OpenCV.
Run: python predict.py
"""

import cv2
import numpy as np
import tensorflow as tf

LABELS = ["Mask", "No Mask"]
COLORS = [(0, 255, 0), (0, 0, 255)]


def detect_masks(model_path="mask_detector.h5"):
    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224)) / 255.0
            face_input = np.expand_dims(face_resized, axis=0)

            preds = model.predict(face_input, verbose=0)[0]
            label_idx = np.argmax(preds)
            label = LABELS[label_idx]
            color = COLORS[label_idx]
            confidence = preds[label_idx] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.1f}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Mask Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_masks()
