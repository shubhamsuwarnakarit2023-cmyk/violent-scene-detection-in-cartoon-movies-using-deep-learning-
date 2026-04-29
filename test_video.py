import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# =========================
# 1️⃣ LOAD TRAINED MODEL
# =========================

with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("best_model1.h5")

print("✅ Model loaded successfully")

# =========================
# 2️⃣ VIDEO INPUT
# =========================

video_path = "normal.mp4"  # must be in same folder
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Cannot open video file")
    exit()

labels = [
    "Violence",
    "Normal",
    "Fight",
    "Explosion",
    "Weapon"
]

SEQUENCE_LENGTH = 8
IMG_SIZE = 224

frames_buffer = []

# =========================
# 3️⃣ READ VIDEO & PREDICT
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_resized = frame_resized.astype("float32") / 255.0

    frames_buffer.append(frame_resized)

    # Predict when buffer is full
    if len(frames_buffer) == SEQUENCE_LENGTH:
        sequence = np.expand_dims(frames_buffer, axis=0)  # (1, 8, 224, 224, 3)

        prediction = model.predict(sequence, verbose=0)   # (1, 8, 5)
        avg_prediction = np.mean(prediction, axis=1)      # (1, 5)

        class_id = np.argmax(avg_prediction, axis=1)[0]
        label = labels[class_id]

        # Show prediction on frame
        cv2.putText(
            frame,
            f"Prediction: {label}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        frames_buffer.pop(0)  # sliding window

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# 4️⃣ CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
