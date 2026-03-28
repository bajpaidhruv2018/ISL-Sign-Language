import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import time
import warnings
import torch
import torch.nn as nn
from collections import deque

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Load label mapping from file and create reverse mapping for decoding predictions
label_mapping = joblib.load('label_mapping.pkl')
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Define the PyTorch network architecture for Indian Sign Language recognition
class ISLNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ISLNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and load pretrained weights
model = ISLNet(42, len(label_mapping)).to(device)
try:
    # Attempt to load with weights_only parameter (new PyTorch versions)
    model.load_state_dict(torch.load('isl_model_pytorch.pth', map_location=device, weights_only=True))
except TypeError:
    # Fallback for older PyTorch versions without weights_only
    model.load_state_dict(torch.load('isl_model_pytorch.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Maintain a fixed length history of predictions to smooth out flickering
prediction_history = deque(maxlen=10)

# Initialize MediaPipe hand landmark detector with hand_landmarker task
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Open the default system webcam
cap = cv2.VideoCapture(0)

# Display program start message
print('\n' + '=' * 50)
print(' ISL Live Terminal Tracker Started ')
print(" Press 'q' in the terminal or video window to stop ")
print('=' * 50 + '\n')

prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate approximate frames per second (FPS) for display
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Convert captured frame to RGB format for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    output_text = f"FPS: {int(fps)} | Waiting for hand sign..."

    if detection_result.hand_landmarks:
        # Extract normalized landmarks relative to the wrist (base landmark)
        hand_coords = []
        base_x = detection_result.hand_landmarks[0][0].x
        base_y = detection_result.hand_landmarks[0][0].y

        for landmark in detection_result.hand_landmarks[0]:
            hand_coords.extend([landmark.x - base_x, landmark.y - base_y])

        # Normalize coordinates to range [-1, 1] based on max absolute value
        max_value = max(abs(coord) for coord in hand_coords)
        if max_value > 0:
            hand_coords = [coord / max_value for coord in hand_coords]

        # Prepare input tensor for the model
        X_test = torch.tensor([hand_coords], dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            conf_value = confidence.item()
            pred_idx = predicted_class.item()

        # Update prediction history if confidence exceeds threshold
        if conf_value > 0.70:  # Confidence threshold set to 70%
            predicted_letter = reverse_mapping[pred_idx]
            prediction_history.append(predicted_letter)
        else:
            prediction_history.append("Unknown")

        # Smooth predictions over recent frames to reduce flickering
        if prediction_history:
            most_common = max(set(prediction_history), key=prediction_history.count)
            if most_common != "Unknown":
                output_text = f"FPS: {int(fps)} | Detected Sign: [ {most_common} ] (Smoothed)"
            else:
                output_text = f"FPS: {int(fps)} | Sign not recognized clearly..."
    else:
        # If no hands detected, remove oldest prediction to avoid stale data
        if prediction_history:
            prediction_history.popleft()

    # Display the status text in the terminal, overwriting previous line
    print(output_text.ljust(80), end='\r')

    # Overlay the status text on the video frame
    cv2.putText(frame, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the live camera feed with overlay
    cv2.imshow('ISL Camera Feed', frame)

    # Exit program if 'q' is pressed in the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources and close all windows upon exit
print('\n\nShutting down camera...')
cap.release()
cv2.destroyAllWindows()
