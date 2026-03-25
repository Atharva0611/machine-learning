import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/plant_disease_mobilenetv2.keras")
LABELS_PATH = Path("models/class_names.json")


@st.cache_resource
def load_model(model_path: Path):
    return tf.keras.models.load_model(model_path)


def load_class_names(labels_path: Path):
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))
    return None


def preprocess_image(image: Image.Image, target_size):
    image = image.convert("RGB").resize(target_size)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def main():
    st.set_page_config(page_title="Plant Disease Detector", layout="centered")
    st.title("🌿 Plant Disease Identification")
    st.write("Upload a plant leaf image to classify disease using MobileNetV2.")

    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Train the model first using: "
            "`python plant_disease_app/train.py --data-dir <your_dataset_path>`"
        )
        return

    model = load_model(MODEL_PATH)
    class_names = load_class_names(LABELS_PATH)

    upload = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if upload is None:
        return

    image = Image.open(upload)
    st.image(image, caption="Uploaded image", use_container_width=True)

    input_height, input_width = model.input_shape[1], model.input_shape[2]
    x = preprocess_image(image, (input_width, input_height))
    preds = model.predict(x)

    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])
    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Class {pred_idx}"

    st.subheader("Prediction")
    st.success(f"**{label}** ({confidence:.2%} confidence)")

    with st.expander("Show probabilities"):
        if class_names and len(class_names) == len(preds[0]):
            probs = {name: float(score) for name, score in zip(class_names, preds[0])}
        else:
            probs = {f"Class {i}": float(score) for i, score in enumerate(preds[0])}
        st.json(probs)


if __name__ == "__main__":
    main()
