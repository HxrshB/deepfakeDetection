import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects, register_keras_serializable
import cv2

# Define and register the FixedDropout class
@register_keras_serializable(package="Custom", name="FixedDropout")
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

# Ensure 'swish' activation is registered
def swish(x):
    return x * tf.keras.activations.sigmoid(x)

get_custom_objects().update({"swish": swish})

# Load the pre-trained model
model_path = "model_checkpoint/best_model.keras"  # Update the path to your model
model = load_model(
    model_path,
    custom_objects={"swish": swish, "FixedDropout": FixedDropout}
)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess frames
def preprocess_frame(frame, target_size=(128, 128)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0  # Normalize pixel values
    return frame

# Function to extract faces from a frame
def extract_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_images = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_images.append(preprocess_frame(face))
    return face_images

# Predict function for a video
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    predictions = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        faces = extract_faces(frame)  # Extract faces from the frame
        for face in faces:
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            prediction = model.predict(face)  # Predict on the face
            predictions.append(prediction[0][0])  # Append the prediction score

    cap.release()

    if len(predictions) == 0:
        print(f"No faces detected in the video.")
        return "No Faces Detected"

    avg_prediction = np.mean(predictions)  # Average over all face predictions
    result = "Real" if avg_prediction >= 0.5 else "Fake"
    print(f"Processed {frame_count} frames, detected {len(predictions)} faces.")
    print(f"Average Prediction: {avg_prediction:.2f}, Result: {result}")
    return result

# Example usage
dataset_path = './test_video/'  # Folder containing videos
for video_name in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video_name)
    if os.path.isfile(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Processing video: {video_name}")
        try:
            result = predict_video(video_path)
            print(f"The video '{video_name}' is {result}.")
        except Exception as e:
            print(f"Error processing video '{video_name}': {e}")
