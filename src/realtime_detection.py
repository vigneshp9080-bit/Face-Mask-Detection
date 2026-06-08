import cv2
import numpy as np
import tensorflow as tf
import json
import os

MODEL_PATH = "model/mobilenet_mask_model.keras"
IMG_SIZE = 224
CLASS_PATH = "model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)
else:
    class_names = ["mask_weared_incorrect", "with_mask", "without_mask"]

print("Loaded classes:", class_names)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Expand face box slightly
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        label = class_names[class_id]
        text = f"{label} {confidence:.2f}%"

        if label == "with_mask":
            color = (0, 255, 0)
        elif label == "without_mask":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Real-Time Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()