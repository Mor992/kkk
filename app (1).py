import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import gdown
# --------------------------------------------------
# Load your trained model
# --------------------------------------------------
MODEL_DRIVE_ID = "1Rw-X-K2o75B70rT2Dwo-WFeA_RvXXWI3"
MODEL_PATH = "resnet_skin_cancer.h5"

st.title("Skin Lesion Classifier")
# ------------------------------
@st.cache_resource
def load_model():
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# --------------------------------------------------
# Class names (update if needed)
# --------------------------------------------------
class_names = ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis"]

# --------------------------------------------------
# Preprocessing function
# --------------------------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------
def generate_gradcam(model, img_array):
    last_conv_layer_name = get_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads = tape.gradient(loss, conv_outputs)

    # Mean intensity of gradients
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Skin Cancer Classification App")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(img)

    # Prediction
    preds = model.predict(img_array)[0]
    pred_index = np.argmax(preds)
    pred_label = class_names[pred_index]

    st.subheader("Prediction")
    st.write(f"**{pred_label}**")

    # Full probabilities
    st.subheader("Probabilities")
    for name, p in zip(class_names, preds):
        st.write(f"{name}: {p:.4f}")

    # Grad-CAM heatmap
    st.subheader("Grad-CAM Heatmap")
    heatmap = generate_gradcam(model, img_array)

    # Combine heatmap + image
    img_np = np.array(img.resize((224, 224)))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    st.image(overlay, caption="Grad-CAM", use_column_width=True)
