import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CUSTOM METRICS & LOSSES
# ============================================================

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (
        (2.0 * intersection + smooth)
        /
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    union = (
        K.sum(y_true_f)
        + K.sum(y_pred_f)
        - intersection
    )

    return (intersection + smooth) / (union + smooth)


bce = BinaryCrossentropy()


def combined_loss(y_true, y_pred):
    return (
        0.5 * bce(y_true, y_pred)
        + 0.5 * dice_loss(y_true, y_pred)
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_segmentation_model():

    model = tf.keras.models.load_model(
        "final_face_segmentation_model.keras",
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou_metric": iou_metric
        }
    )

    return model


model = load_segmentation_model()


# ============================================================
# UI
# ============================================================

st.title("Face Segmentation App")
st.write("Upload an image and the model will predict the face mask.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # PREPROCESS
    # ========================================================

    input_image = cv2.resize(
        image_rgb,
        (128, 128)
    )

    input_image = preprocess_input(
        input_image.astype(np.float32)
    )



    input_image = np.expand_dims(
        input_image,
        axis=0
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    pred_mask = model.predict(
        input_image,
        verbose=0
    )[0]

  

    pred_mask = pred_mask.squeeze()

    # Resize mask to original image size
    pred_mask = cv2.resize(
        pred_mask,
        (
            image_rgb.shape[1],
            image_rgb.shape[0]
        )
    )

    # Convert to visualization image
    pred_mask_vis = (pred_mask > 0.5).astype(np.uint8) * 255

    # ========================================================
    # DISPLAY
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image_rgb,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            pred_mask_vis,
            caption="Predicted Mask",
            use_container_width=True,
            clamp=True
        )

    st.success("Prediction completed!")