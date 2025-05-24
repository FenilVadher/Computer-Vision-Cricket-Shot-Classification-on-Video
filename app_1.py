import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, Input
from tensorflow.keras.applications import EfficientNetB0
import tempfile
import shutil

st.set_page_config(layout="wide")

# Define class labels
classes = {'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 'late_cut': 4, 'lofted': 5, 'pull': 6, 'square_cut': 7, 'straight': 8, 'sweep': 9}

# Function to load the model
def load_model(weights_path):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    video_input = Input(shape=(None, 224, 224, 3))
    x = layers.TimeDistributed(base_model)(video_input)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.GRU(256, return_sequences=True)(x)
    x = layers.GRU(128)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=video_input, outputs=output)
    model.load_weights(weights_path)
    return model

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame.numpy()

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=1):
    result = []
    src = cv2.VideoCapture(str(video_path))

    src.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = src.read()
    if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
    else:
        result.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

def classify_video(video_path, model, frame_count, class_labels):
    frames = frames_from_video_file(video_path, frame_count)
    frames = np.expand_dims(frames, axis=0)
    predictions = model.predict(frames)

    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name = list(class_labels.keys())[list(class_labels.values()).index(predicted_class_idx)]
    confidence = predictions[0][predicted_class_idx] * 100
    return predicted_class_name, confidence

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmpfile:
            shutil.copyfileobj(uploaded_file, tmpfile)
            return tmpfile.name
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return None

# Streamlit UI
st.title('Cricket Shot Classification and Similarity Checker')

# Load model 
model = load_model('model_weights.h5')

col1, col2 = st.columns(2)
class1 = conf1 = class2 = conf2 = None
video1_path = video2_path = None

with col1:
    video1 = st.file_uploader("Upload first video", type=["mp4", "avi"], key="video1")
    if video1:
        st.video(video1)
        video1_path = save_uploaded_file(video1)
        class1, conf1 = classify_video(video1_path, model, 30, classes)
        st.success(f"First video classified as {class1} with confidence {conf1:.2f}%")

with col2:
    video2 = st.file_uploader("Upload second video", type=["mp4", "avi"], key="video2")
    if video2:
        st.video(video2)
        video2_path = save_uploaded_file(video2)
        class2, conf2 = classify_video(video2_path, model, 30, classes)
        st.success(f"Second video classified as {class2} with confidence {conf2:.2f}%")

if st.button('Compare Videos'):
    if video1_path and video2_path and class1 == class2:
        # Feature extraction model (output from GRU before dense layers)
        feature_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)

        frames1 = np.expand_dims(frames_from_video_file(video1_path, 30), axis=0)
        frames2 = np.expand_dims(frames_from_video_file(video2_path, 30), axis=0)

        features1 = feature_model.predict(frames1)
        features2 = feature_model.predict(frames2)

        dot_product = np.dot(features1, features2.T)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        similarity = dot_product / (norm1 * norm2)

        st.info(f"Cosine Similarity: {similarity[0][0]:.4f}")
    else:
        st.warning("Upload both videos and ensure they belong to the same class to compare.")
