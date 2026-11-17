
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from matplotlib import colormaps


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    APP_DIR, "pneumovision_cnn_best.keras"
)

THRESHOLD_PATH = os.path.join(
    APP_DIR, "threshold_config.json"
)

IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "conv2d_9"

st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LOAD MODEL AND CONFIGURATION
# ============================================================

@st.cache_resource
def load_pneumovision_model():

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    gradcam_model = tf.keras.Model(
        inputs=loaded_model.inputs,
        outputs=[
            loaded_model.get_layer(
                LAST_CONV_LAYER
            ).output,
            loaded_model.output
        ]
    )

    with open(THRESHOLD_PATH, "r") as file:
        config = json.load(file)

    threshold = float(
        config["classification_threshold"]
    )

    return loaded_model, gradcam_model, threshold


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_image):

    # Match training preprocessing:
    # grayscale, resize 224x224, normalize to 0-1

    image = uploaded_image.convert("L")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array, image_batch


# ============================================================
# 4. GRAD-CAM GENERATION
# ============================================================

def generate_gradcam(gradcam_model, image_batch):

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = gradcam_model(
            image_tensor,
            training=False
        )

        pneumonia_score = predictions[:, 0]

    gradients = tape.gradient(
        pneumonia_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + 1e-8),
        tf.zeros_like(heatmap)
    )

    return heatmap.numpy()


# ============================================================
# 5. CREATE HEATMAP OVERLAY
# ============================================================

def create_overlay(image_array, heatmap):

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    )

    heatmap_image = heatmap_image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    heatmap_resized = (
        np.asarray(heatmap_image).astype(
            np.float32
        ) / 255.0
    )

    colored_heatmap = (
        colormaps["jet"](heatmap_resized)[:, :, :3]
        * 255
    ).astype(np.uint8)

    original = (
        np.squeeze(image_array) * 255
    ).clip(0, 255).astype(np.uint8)

    original_rgb = Image.fromarray(
        original
    ).convert("RGB")

    heatmap_rgb = Image.fromarray(
        colored_heatmap
    )

    overlay = Image.blend(
        original_rgb,
        heatmap_rgb,
        alpha=0.40
    )

    return heatmap_rgb, overlay


# ============================================================
# 6. APPLICATION HEADER
# ============================================================

st.title("🫁 PneumoVision AI")

st.subheader(
    "Deep Learning-Based Pneumonia Detection "
    "from Chest X-Ray Images"
)

st.write(
    "An educational deep learning prototype using a "
    "custom Convolutional Neural Network (CNN) and "
    "Grad-CAM explainability."
)

st.divider()


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("About the Project")

    st.write(
        "PneumoVision AI uses a custom CNN trained on "
        "chest X-ray images to classify images into "
        "Normal and Pneumonia categories."
    )

    st.markdown("### Model information")

    st.write("**Architecture:** Custom CNN")
    st.write("**Input size:** 224 × 224")
    st.write("**Input channels:** Grayscale")
    st.write("**Explainability:** Grad-CAM")
    st.write("**Output classes:** Normal / Pneumonia")

    st.divider()

    st.caption(
        "Academic project prototype — not intended "
        "for medical diagnosis."
    )


# ============================================================
# 8. LOAD MODEL
# ============================================================

try:

    model, gradcam_model, threshold = (
        load_pneumovision_model()
    )

    st.success("CNN model loaded successfully.")

except Exception as error:

    st.error(
        "The model could not be loaded. "
        "Please check the model and configuration files."
    )

    st.exception(error)
    st.stop()


# ============================================================
# 9. IMAGE UPLOAD
# ============================================================

st.header("Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear chest X-ray image in JPG or PNG format."
)

if uploaded_file is None:

    st.info(
        "Upload a chest X-ray image to begin the analysis."
    )

    st.stop()


# ============================================================
# 10. DISPLAY UPLOADED IMAGE
# ============================================================

try:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception:

    st.error("The uploaded file is not a valid image.")
    st.stop()


st.subheader("Uploaded Image")

st.image(
    original_image,
    caption="Uploaded chest X-ray",
    width=450
)


# ============================================================
# 11. PREDICTION
# ============================================================

if st.button(
    "Analyze X-ray",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Processing image and generating prediction..."
    ):

        processed_image, image_batch = (
            preprocess_image(original_image)
        )

        prediction = model.predict(
            image_batch,
            verbose=0
        )

        pneumonia_probability = float(
            prediction[0][0]
        )

        normal_probability = (
            1.0 - pneumonia_probability
        )

        predicted_class = (
            "PNEUMONIA"
            if pneumonia_probability >= threshold
            else "NORMAL"
        )

        heatmap = generate_gradcam(
            gradcam_model,
            image_batch
        )

        heatmap_image, overlay = create_overlay(
            processed_image,
            heatmap
        )


    # --------------------------------------------------------
    # 12. DISPLAY PREDICTION
    # --------------------------------------------------------

    st.divider()
    st.header("Prediction Results")

    result_col, confidence_col = st.columns(2)

    with result_col:

        if predicted_class == "PNEUMONIA":

            st.error("Prediction: PNEUMONIA")

        else:

            st.success("Prediction: NORMAL")

    with confidence_col:

        st.metric(
            "Pneumonia probability",
            f"{pneumonia_probability * 100:.2f}%"
        )

    st.write(
        f"**Normal probability:** "
        f"{normal_probability * 100:.2f}%"
    )

    st.write(
        f"**Classification threshold:** {threshold:.4f}"
    )

    st.progress(
        float(np.clip(pneumonia_probability, 0, 1))
    )

    st.caption(
        "The displayed probability is the model's output, "
        "not a clinically calibrated probability."
    )


    # --------------------------------------------------------
    # 13. GRAD-CAM EXPLAINABILITY
    # --------------------------------------------------------

    st.divider()
    st.header("Explainable AI — Grad-CAM")

    st.write(
        "The heatmap visualizes image regions that "
        "positively influenced the model's Pneumonia score."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            heatmap_image,
            caption="Grad-CAM heatmap",
            use_container_width=True
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM overlay",
            use_container_width=True
        )


    # --------------------------------------------------------
    # 14. INTERPRETATION
    # --------------------------------------------------------

    st.info(
        "Warmer colours indicate regions with stronger "
        "positive influence on the model's Pneumonia score. "
        "These regions are not confirmed disease locations."
    )

    st.divider()

    st.subheader("Important Disclaimer")

    st.warning(
        "PneumoVision AI is an academic research prototype. "
        "It is not a medical device and must not be used "
        "for diagnosis, treatment decisions, or emergency "
        "medical decisions. Predictions may be incorrect. "
        "Consult a qualified medical professional for "
        "medical assessment."
    )

    st.caption(
        "PneumoVision AI | CNN + Grad-CAM | "
        "Academic Neural Network and Deep Learning Project"
    )
