import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from auth import *
from history import *

# -------------------------
# DB Setup
# -------------------------
create_user_table()
create_history()

st.set_page_config(layout="wide", page_title="Leaf Disease AI")

# -------------------------
# Session Management
# -------------------------
if "login" not in st.session_state:
    st.session_state.login = False

# -------------------------
# Disease Classes
# -------------------------
classes = [
    "Healthy",
    "Powdery_Mildew",
    "Rust",
    "Leaf_Spot",
    "Early_Blight",
    "Late_Blight",
    "Septoria_Leaf_Spot",
    "Downy_Mildew",
    "Anthracnose",
    "Fusarium_Wilt",
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
]

disease_desc = {
    "Healthy": "Leaf is healthy with no visible signs of disease.",
    "Powdery_Mildew": "White powdery fungal growth on leaves, stems, or buds.",
    "Rust": "Small reddish-brown or orange pustules on leaf surfaces caused by fungal spores.",
    "Leaf_Spot": "Dark circular spots with yellow halos caused by fungi or bacteria.",
    "Early_Blight": "Concentric dark spots appearing on older leaves of plants.",
    "Late_Blight": "Brown lesions with white fungal growth on leaf surface.",
    "Septoria_Leaf_Spot": "Small dark spots with light centers, often on older leaves.",
    "Downy_Mildew": "Yellow patches on leaves with downy growth underneath.",
    "Anthracnose": "Dark sunken lesions on leaves, stems, or fruits.",
    "Fusarium_Wilt": "Wilting and yellowing due to vascular infection.",
    "Bacterial_Spot": "Small water-soaked lesions on leaves that turn brown.",
    "Cercospora_Leaf_Spot": "Brown or gray circular spots with dark borders.",
}

disease_prev = {
    "Healthy": "Maintain proper watering, sunlight, and nutrients. Monitor regularly.",
    "Powdery_Mildew": "Apply neem oil or sulfur-based fungicides. Improve air circulation.",
    "Rust": "Remove infected leaves. Use resistant varieties. Apply fungicides if needed.",
    "Leaf_Spot": "Avoid overwatering. Remove infected leaves. Use copper fungicides.",
    "Early_Blight": "Remove infected leaves, rotate crops, and use fungicides.",
    "Late_Blight": "Apply appropriate fungicides and avoid overhead watering.",
    "Septoria_Leaf_Spot": "Remove old infected leaves, use disease-resistant varieties.",
    "Downy_Mildew": "Improve ventilation and use fungicides for infected crops.",
    "Anthracnose": "Remove infected parts and apply fungicides if necessary.",
    "Fusarium_Wilt": "Use resistant varieties and crop rotation to prevent spread.",
    "Bacterial_Spot": "Remove infected leaves and avoid wetting foliage.",
    "Cercospora_Leaf_Spot": "Apply fungicides and ensure proper spacing for air circulation.",
}


# -------------------------
# Load Model (cached)
# -------------------------
@st.cache_resource
def load_my_model(path="model.keras"):
    if not os.path.exists(path):
        st.error(
            f"❌ {path} not found. Please train and place the model in the folder."
        )
        st.stop()
    return load_model(path)


model = load_my_model()


# -------------------------
# Leaf Detection / Segmentation
# -------------------------
def is_leaf(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
    return ratio > 0.05, mask


def segment_leaf(img, mask):
    leaf = cv2.bitwise_and(img, img, mask=mask)
    return leaf


# -------------------------
# Login Page
# -------------------------
def login_page():
    st.markdown(
        "<h1 style='text-align:center'>🌿 Leaf Disease Detection</h1>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("🔐 Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            data = login_user(user, pwd)
            if data:
                st.session_state.login = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("---")
        st.subheader("🆕 Register")
        new_user = st.text_input("New Username")
        new_pwd = st.text_input("New Password", type="password")

        if st.button("Create Account", use_container_width=True):
            add_user(new_user, new_pwd)
            st.success("Account created")


# -------------------------
# Main App
# -------------------------
def main():
    st.sidebar.title("🌿 Dashboard")
    st.sidebar.success(f"User: {st.session_state.user}")
    st.sidebar.info("Model: Fine-tuned MobileNetV2")

    page = st.sidebar.selectbox("Menu", ["Detect", "History"])

    if page == "Detect":
        st.title("🌿 Leaf Disease Detection")
        col1, col2 = st.columns(2)

        with col1:
            plant = st.selectbox("Plant Type", ["Tomato", "Potato", "Rice", "Wheat"])
            st.markdown("**Upload Leaf Image**")
            file = st.file_uploader("Upload Leaf Image", type=["png", "jpg", "jpeg"])
            st.markdown("**Or Take Photo from Camera**")
            cam_file = st.camera_input("Capture Leaf Image")

        # Choose uploaded or camera image
        img_source = file if file is not None else cam_file

        if img_source:
            image = Image.open(img_source)
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            with col2:
                st.image(image, width=350)

            # Leaf detection
            leaf_found, mask = is_leaf(img)
            if not leaf_found:
                st.error("❌ Please upload a leaf image only")
                return
            st.success("✅ Leaf detected")
            leaf_img = segment_leaf(img, mask)

            if st.button("Analyze", use_container_width=True):
                # Preprocess for MobileNetV2
                im = cv2.resize(leaf_img, (224, 224))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = np.expand_dims(im, axis=0)
                im = preprocess_input(im)

                pred = model.predict(im)
                idx = np.argmax(pred)
                conf = float(np.max(pred)) * 100
                disease = classes[idx]

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Disease", disease)
                with c2:
                    st.metric("Confidence", f"{conf:.2f}%")
                st.progress(int(conf))

                st.subheader("📖 Description")
                st.info(disease_desc[disease])

                st.subheader("🛡 Prevention")
                with st.expander("View Prevention Tips"):
                    st.success(disease_prev[disease])

                # Save history
                add_history(st.session_state.user, plant, disease, conf)

    if page == "History":
        st.title("📈 Prediction History")
        data = get_history(st.session_state.user)
        for row in data:
            st.write(row)


# -------------------------
# Run App
# -------------------------
if not st.session_state.login:
    login_page()
else:
    main()
