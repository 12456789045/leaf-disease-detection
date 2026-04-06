#### PYTHON|STREAMLIT|LICENSE|MIT


# 🌿 Leaf Disease Detection AI

An AI-powered **Leaf Disease Detection Web App** built using **Streamlit** and **TensorFlow MobileNetV2**.
Users can upload or capture leaf images, detect plant diseases, and store prediction history with authentication.

---

## 🚀 Features

* 🔐 User Login & Registration
* 🌿 Leaf Detection using OpenCV
* 🤖 AI Disease Classification (MobileNetV2)
* 📷 Upload Image OR Use Camera
* 📊 Confidence Score Display
* 📖 Disease Description
* 🛡 Prevention Tips
* 📈 Prediction History
* 🗄 Local Database (No external DB)
* 📱 Mobile Responsive Streamlit UI

---

## 🛠 Tech Stack

* Streamlit
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Pillow
* SQLite (Local)
* Python

---

## 📂 Project Structure

```
leaf-disease-ai/
│
├── app.py
├── model.keras
├── auth.py
├── history.py
├── README.md
└── requirements.txt
```

---

## 📦 Installation

### 1. Clone / Create Project

```
mkdir leaf-disease-ai
cd leaf-disease-ai
```

### 2. Create Virtual Environment

```
python -m venv venv
```

Activate:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 3. Install Dependencies

```
pip install streamlit tensorflow opencv-python numpy pillow pandas
```

---

## ▶️ Run Application

```
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 🧠 AI Model

The application uses **Fine-tuned MobileNetV2** trained on plant leaf dataset.

Supported Classes:

* Healthy
* Powdery_Mildew
* Rust
* Leaf_Spot

---

## 📸 How It Works

1. User logs in / registers
2. Upload leaf image OR capture using camera
3. System detects leaf using segmentation
4. Image processed for MobileNetV2
5. AI predicts disease
6. Confidence score shown
7. Description & prevention tips displayed
8. Result stored in history

---

## 📈 History Tracking

Each prediction stores:

* Username
* Plant Type
* Disease
* Confidence
* Timestamp

---

## 🛡 Leaf Detection

The system uses:

* Grayscale conversion
* Gaussian blur
* Threshold segmentation
* Mask extraction

This ensures **only leaf images** are processed.

---

## ⚠️ Important Notes

* `model.keras` must be present in project folder
* Model output classes must match code
* Data stored locally (SQLite)
* No internet required
* Works offline

---

## 📱 Supported Input

* PNG
* JPG
* JPEG
* Camera Capture

---

## 👤 Authentication

* Register new user
* Login existing user
* Session-based access
* User-specific history

---

## ⭐ Future Improvements

* Multi-plant disease dataset
* Heatmap visualization
* Severity detection
* Treatment recommendation AI
* Export history CSV
* Admin dashboard

---

## 👨‍💻 Author

Hrishikesh Kulkarni
python|FastAPI|Streamlit|AI Developer

---
