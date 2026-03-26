# Indian Sign Language (ISL) Real-Time Recognition 🇮🇳

An **Open Innovation** project for AI/ML that recognizes Indian Sign Language (ISL) alphabets (A-Z) and numbers (1-9) in real-time using **Computer Vision** and **Machine Learning**.

## 🚀 Project Overview
This project addresses the communication gap for the hearing and speech-impaired community. Instead of using heavy CNN models, this implementation uses **MediaPipe** to extract hand landmarks (skeleton) and a **Random Forest Classifier** for high-speed, real-time inference directly in the terminal.

## 📊 Dataset Information
The model is trained using the **Indian Sign Language (ISL)** dataset by Prathum Arikeri.
* **Source:** [Kaggle - Indian Sign Language (ISL)](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
* **Original Scale:** 40,000+ images across 35 classes (A-Z, 1-9).
* **Augmentation:** To improve real-world accuracy, the dataset was expanded to **228,000+ augmented samples** (covering various rotations, scales, and lighting conditions).



## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Core Libraries:** PyTorch, Scikit-learn, NumPy, Pandas
- **Computer Vision:** OpenCV, MediaPipe
- **Development Tool:** Google Antigravity (Agentic AI Workflow)
- **Version Control:** Git & GitHub

## 📁 Project Structure
```text
.
├── data/               # Folder containing A-Z, 1-9 image folders (Ignored by Git)
├── data_builder.py     # Script to extract landmarks and train the model
├── run_isl.py          # Script for real-time terminal-based recognition
├── isl_model.pkl       # The trained model weights (Ignored by Git)
├── README.md           # Documentation
└── .gitignore          # Prevents large data/weights from being uploaded

⚙️ Setup and Execution1. Clone the RepositoryBashgit clone [https://github.com/bajpaidhruv2018/ISL-Sign-Language.git](https://github.com/bajpaidhruv2018/ISL-Sign-Language.git)
cd ISL-Sign-Language
2. Download DataDownload the dataset from Kaggle and place the subfolders (A, B, C...) inside a directory named data/.3. Training the ModelRun this script to process the 228k+ samples and generate the model file:Bashpython data_builder.py
4. Real-Time InferenceStart the webcam-based terminal recognition:Bashpython run_isl.py
🧠 Why This Works (Open Innovation)Coordinate-Based Learning: By focusing on the 21 $(x, y)$ hand joints (42 total features), the model ignores background noise and focuses purely on hand geometry.Efficiency: The Random Forest model allows for 30+ FPS performance on standard hardware (like an RTX 4050), making it viable for edge devices.Robustness: Using a 70% Confidence Threshold, the system avoids misclassifying random hand movements, ensuring high-quality predictions.

---

