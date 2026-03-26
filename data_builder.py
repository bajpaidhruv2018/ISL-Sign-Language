import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# imported media pipe and hand tracing api 
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def train_isl_model(dataset_folder):
    print("Starting landmark extraction. This might take a few minutes...")
    extracted_data = []
    
    # looping thorugh data set , usinf forr loop to analyse the data set
    for sign_label in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, sign_label)
        
        # check for dictionary and skiping, finding paths from dictionary
        if not os.path.isdir(folder_path):
            continue
            
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)
            
            # finding coordinates for hand gestures
            if detection_result.hand_landmarks:
                hand_coords = []
                base_x = detection_result.hand_landmarks[0][0].x
                base_y = detection_result.hand_landmarks[0][0].y
                
                for landmark in detection_result.hand_landmarks[0]:
                    hand_coords.extend([landmark.x - base_x, landmark.y - base_y])
                
                # finidng distance -1 to 1 , using to measure coordinate for hands
                max_value = max([abs(x) for x in hand_coords])
                if max_value > 0:
                    hand_coords = [x / max_value for x in hand_coords]
                
                # showing hand sign, diplsying hand sign from user
                hand_coords.append(sign_label)
                extracted_data.append(hand_coords)

    print(f"Extraction complete! Found {len(extracted_data)} original hand images.")
    
    #
    # increasing dataset by 10 times, it using massive data set for training
    #
    print("Augmenting data to create a massively larger dataset...")
    augmented_data = []
    
    for row in extracted_data:
        features = row[:-1]
        label = row[-1]
        
        # main sample , its the original copt
        augmented_data.append(row)
        
        # creating 9 variation
        for _ in range(9):
            # adding gaussian noise to simulate finger/jitter so that we could analyse the hand signs
            noisy_features = [f + np.random.normal(0, 0.02) for f in features]
            noisy_row = noisy_features + [label]
            augmented_data.append(noisy_row)
            
    print(f"Dataset successfully expanded to {len(augmented_data)} samples!")
    print("Using 100% RTX GPU Power: Training PyTorch Deep Learning Model...")
    
    # conversion to data frames , so that i could be displayed
    df = pd.DataFrame(augmented_data)
    X_features = df.iloc[:, :-1].values.astype('float32')
    y_labels_str = df.iloc[:, -1].values
    
    # conversion labels to integers
    unique_labels = sorted(list(set(y_labels_str)))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_labels = [label_mapping[label] for label in y_labels_str]
    
    # mapping and taking coordinates
    joblib.dump(label_mapping, 'label_mapping.pkl')
    
    # setting up PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n====================================")
    print(f" DEVICE SELECTED FOR TRAINING: {device}")
    print(f"====================================\n")
    
    X_tensor = torch.tensor(X_features).to(device)
    y_tensor = torch.tensor(y_labels, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Deep neural network architecture
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
            
    num_classes = len(unique_labels)
    classifier = ISLNet(X_features.shape[1], num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    epochs = 40
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == batch_y).sum().item()
        
        acc = 100 * correct / len(dataset)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {acc:.2f}%")
            
    # saving PyTorch model it saving model for work
    torch.save(classifier.state_dict(), 'isl_model_pytorch.pth')
    print("Success: PyTorch GPU model trained and saved as 'isl_model_pytorch.pth'!")

# running function assuming images in data it recursive call to run function
train_isl_model('data')