import os
import json
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import openai
import h5py

# Streamlit App Title
st.title("🌿 Plant Disease Classifier & Chatbot Assistant")

# ✅ User input for OpenAI API Key
api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# Only set API key if user has entered it
if api_key:
    openai.api_key = api_key
else:
    st.warning("⚠️ Please enter a valid OpenAI API Key to use AI features.")

# ✅ Set working directory and model paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "app", "trained_model")  # Updated model path
model_h5_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")
model_keras_path = os.path.join(model_dir, "plant_disease_prediction_model.keras")

# ✅ Google Drive Model Download URL
gdrive_file_id = "1WLJk_JlWYL-1M8enmRgiCx3ddYNJwDUv"
gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

# ✅ Ensure the trained_model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ✅ Check if the .h5 model exists, else download from Google Drive
if not os.path.exists(model_h5_path):
    st.warning("⚠️ Model file not found locally. Downloading from Google Drive...")
    try:
        gdown.download(gdrive_url, model_h5_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")
        model_h5_path = None  # Prevent crashing if model download fails

# ✅ Check if the .h5 model is valid before conversion
if model_h5_path and os.path.exists(model_h5_path):
    try:
        with h5py.File(model_h5_path, "r") as f:
            st.success("✅ Model file is valid and readable!")
    except Exception as e:
        st.error(f"❌ Corrupt model file: {e}")
        model_h5_path = None  # Prevent further crashes

# ✅ Convert .h5 model to .keras format if needed
if model_h5_path and os.path.exists(model_h5_path) and not os.path.exists(model_keras_path):
    try:
        st.warning("🔄 Converting model from .h5 to .keras format for better compatibility...")
        model = tf.keras.models.load_model(model_h5_path)
        model.save(model_keras_path, save_format="keras")
        st.success("✅ Model successfully converted to .keras format!")
    except Exception as e:
        st.error(f"❌ Error converting model: {e}")
        model_keras_path = None  # Prevent crashes

# ✅ Load the model from .keras format
model = None
if model_keras_path and os.path.exists(model_keras_path):
    try:
        model = tf.keras.models.load_model(model_keras_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ Model file is missing. Please check the Google Drive link or upload the model manually.")

# ✅ Load class names
class_indices_path = os.path.join(working_dir, "app", "class_indices.json")
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
    class_indices = {int(k): v for k, v in class_indices.items()}
else:
    st.error("❌ Class indices file missing. Please check your project folder.")
    class_indices = {}

# ✅ Function to Load and Preprocess Image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# ✅ Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    if model is None:
        return "No Model", 0.0

    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(predicted_class_index, "Unknown Disease")
    confidence_score = np.max(predictions) * 100
    return predicted_class_name, confidence_score

# ✅ Function to Get Disease Description from GPT-4
def get_disease_description(disease_name):
    if not api_key:
        return "⚠️ API key is missing. Please enter a valid API key above."

    prompt = f"""Provide a detailed description of the plant disease '{disease_name}'. Include:
    1. A brief description of the disease.
    2. Possible causes and environmental conditions leading to it.
    3. Treatment and preventive measures.
    4. Associated plant species that may also be affected.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in plant diseases."},
                      {"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Function for GPT-4 Chatbot
def chatbot_response(user_query):
    if not api_key:
        return "⚠️ API key is missing. Please enter a valid API key above."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system",
                       "content": "You are a plant expert. Answer user queries about plant diseases and plant care."},
                      {"role": "user", "content": user_query}],
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Create Tabs
tab1, tab2 = st.tabs(["🖼 Disease Detection", "💬 Chat with AI"])

# 🖼 **Tab 1: Plant Disease Detection**
with tab1:
    st.subheader("📸 Capture or Upload Images for Disease Detection")

    # **Real-time Camera Upload**
    camera_image = st.camera_input("📷 Take a photo of a plant leaf")

    # **Batch Image Upload**
    uploaded_images = st.file_uploader("📤 Upload plant images (Multiple Allowed)", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True)

    images_to_process = []

    # Add camera image if taken
    if camera_image:
        images_to_process.append(Image.open(camera_image))

    # Add uploaded images
    if uploaded_images:
        for img_file in uploaded_images:
            images_to_process.append(Image.open(img_file))

    # **Process Multiple Images**
    if images_to_process and model:
        st.subheader("🖼 Processed Images & Results")

        for idx, image in enumerate(images_to_process):
            col1, col2 = st.columns([1, 2])

            with col1:
                resized_img = image.resize((200, 200))
                st.image(resized_img, caption=f"Image {idx + 1}", use_column_width=True)

            with col2:
                predicted_disease, confidence = predict_image_class(model, image, class_indices)
                disease_details = get_disease_description(predicted_disease)

                st.success(f"🌱 **Prediction:** {predicted_disease}")
                st.info(f"🔢 **Confidence:** {confidence:.2f}%")
                st.markdown(f"📖 **Disease Information:**\n\n{disease_details}")
