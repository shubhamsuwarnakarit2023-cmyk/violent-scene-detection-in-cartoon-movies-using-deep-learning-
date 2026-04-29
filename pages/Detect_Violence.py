import streamlit as st
import numpy as np
import cv2
import tempfile
import tensorflow as tf

# Load model with caching (better performance)
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model("video_model.keras")

model = load_model_cached()

class_names = ['bloody', 'explosions', 'fight', 'gun shot', 'non-violence', 'violence']

st.title("🎬 Violent Scene Detection System")
st.write("Upload a video file to detect violent scenes.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return None

    for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        return None

    return np.expand_dims(frames, axis=0)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Detect Violence"):
        input_data = extract_frames(tfile.name)

        if input_data is None:
            st.error("Video is too short or invalid.")
        else:
            prediction = model.predict(input_data)

            if prediction.ndim == 3:
                prediction = np.mean(prediction, axis=1)

            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            st.success(f"Prediction: {class_names[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}%")