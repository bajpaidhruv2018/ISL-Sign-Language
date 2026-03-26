import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import time
import warnings
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

warnings.filterwarnings('ignore')

label_mapping = joblib.load('label_mapping.pkl')
reverse_mapping = {v: k for k, v in label_mapping.items()}

# PyTorch network matching the training side
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ISLNet(42, len(label_mapping)).to(device)
try:
    model.load_state_dict(torch.load('isl_model_pytorch.pth', map_location=device, weights_only=True))
except TypeError:
    # Fallback for older PyTorch versions
    model.load_state_dict(torch.load('isl_model_pytorch.pth', map_location=device))
model.eval()

# To stop the blinking/flickering, we use a smoothing window of 10 frames
prediction_history = deque(maxlen=10)

# Setup MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Open default webcam
cap = cv2.VideoCapture(0)

print("\n" + "="*50)
print(" ISL Live Terminal Tracker Started ")
print(" Press 'q' in the terminal or video window to stop ")
print("="*50 + "\n")

prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    output_text = f"FPS: {int(fps)} | Waiting for hand sign..."
    
    if detection_result.hand_landmarks:
        hand_coords = []
        base_x = detection_result.hand_landmarks[0][0].x
        base_y = detection_result.hand_landmarks[0][0].y
        
        for landmark in detection_result.hand_landmarks[0]:
            hand_coords.extend([landmark.x - base_x, landmark.y - base_y])
            
        # Normalize distances to be between -1 and 1
        max_value = max([abs(x) for x in hand_coords])
        if max_value > 0:
            hand_coords = [x / max_value for x in hand_coords]
        
        X_test = torch.tensor([hand_coords], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_test)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            highest_confidence, predicted_class = torch.max(probabilities, 1)
            
            conf_value = highest_confidence.item()
            pred_idx = predicted_class.item()
            
        if conf_value > 0.70: # 70% threshold
            predicted_letter = reverse_mapping[pred_idx]
            prediction_history.append(predicted_letter)
        else:
            prediction_history.append("Unknown")
            
        # Smoothing: Most common prediction in last 10 frames
        if len(prediction_history) > 0:
            most_common = max(set(prediction_history), key=prediction_history.count)
            if most_common != "Unknown":
                output_text = f"FPS: {int(fps)} | Detected Sign: [ {most_common} ] (Smoothed)"
            else:
                output_text = f"FPS: {int(fps)} | Sign not recognized clearly..."
    else:
        # Clear history slowly if no hand is shown to prevent sticky signs
        if len(prediction_history) > 0:
            prediction_history.pop()

    # Print to terminal on the same line
    print(output_text.ljust(80), end='\r')

    # Add the output text to the video frame
    cv2.putText(frame, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # We uncomment this to show the camera window
    cv2.imshow('ISL Camera Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n\nShutting down camera...")
cap.release()
cv2.destroyAllWindows()