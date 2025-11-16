import io
import os
import pathlib
import pickle
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn import preprocessing
from tensorflow import keras

from utils import *  # Ensure this file exists in your directory

st.set_page_config(layout="wide")

data_dir = pathlib.Path("C:\\Users\\krnps\\Downloads\\Deep-learning-recommender-main\\Deep-learning-recommender-main\\data")

IMG_SIZE = (224, 224)
batch_size = 50
preprocess_input = tf.keras.applications.vgg16.preprocess_input
weights_file = "C:\\Users\\krnps\\MODERN\\vgg16_furniture_classifier_1129.h5"

# Load VGG16 model
base_model = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(6, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
)
model.load_weights(weights_file)

st.title("Furniture Collection Recommender")
st.markdown(
    """
    This is an image-based product collection recommender that pairs user-inputted products with 
    other visually compatible furniture collections. Use the sidebar to select the type of furniture.
    """
)

st.sidebar.subheader("Pick the type of furniture you want to be recommended")
ref_option = st.sidebar.selectbox(
    "Choose furniture type:",
    ('chair', 'couch', 'clock', 'plant_pot', 'table', 'bed'),
    key="ref",
)

Data_root = "C:\\Users\\krnps\\MODERN\\pcl"
ref_path_dir = os.path.join(Data_root, ref_option)
all_reg_path = [
    os.path.join(ref_path_dir, fname) for fname in sorted(os.listdir(ref_path_dir))
]

# Ensure user enters a valid number
ref_id = st.sidebar.text_input("Enter number of recommendations", "5")
if not ref_id.isnumeric():
    st.sidebar.error("Please enter a valid number")
    ref_id = 5
else:
    ref_id = int(ref_id)


def similarity_search(V, v_query, file_mapping, n_results=ref_id + 1):
    v_query = np.expand_dims(v_query, axis=0)
    d = V.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(V))
    distances, closest_indices = index.search(v_query, n_results)
    distances = distances.flatten()
    closest_indices = closest_indices.flatten()
    closest_paths = [file_mapping[idx] for idx in closest_indices]
    results_img = get_concatenated_images(closest_paths)
    return closest_paths, results_img


def image_upload(img, target_size):
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


fc1_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("fc1").output)
fc2_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("fc2").output)

file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(file)
    img_show = img.resize((288, 288))
    compare = preprocessing.normalize(
        fc1_extractor.predict(image_upload(img, model.input_shape[1:3])[1]), norm="l2",
    ).reshape(4096,)

    closest_paths, results = similarity_search(
        pickle.load(open(all_reg_path[0], "rb")),
        compare,
        pickle.load(open(all_reg_path[1], "rb")),
    )

    st.subheader("Uploaded Furniture")
    st.image(img_show, width=None)

    # Only show images (no price or website link)
    st.subheader("Recommended Furniture")
    for k in range(len(closest_paths) - 1):
        st.image(get_concatenated_images([closest_paths[k + 1]]), caption=f"Recommendation {k + 1}")

import subprocess

def is_adb_connected():
    """Check if ADB detects an Android device."""
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
        devices = result.stdout.splitlines()[1:]  # Skip first line (header)
        return any("device" in line for line in devices)  # Look for "device" status
    except Exception as e:
        return False


if file is not None and 'closest_paths' in locals():
    # Extract recommended furniture model name
    recommended_model = os.path.splitext(os.path.basename(closest_paths[1]))[0]
    ar_link = f"furnai://open?model={recommended_model}"  # Create deep link

    st.subheader("🔗 View in AR")
    if st.button("Open AR Viewer 📱"):
        if is_adb_connected():
            subprocess.run(["adb", "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", ar_link])
            st.success("AR Viewer launched on your mobile device! ✅")
        else:
            st.error("No device found! Connect your phone via USB or WiFi and enable ADB.")   
