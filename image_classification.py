import streamlit as st
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image as tf_image
import numpy as np
from PIL import Image
from io import BytesIO
from PIL import UnidentifiedImageError

# Load the pre-trained MobileNetV2 model + higher level layers
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))
])

# Load the labels file for ImageNet
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
imagenet_labels = np.array(open(labels_path).read().splitlines())

def load_image(img):
    img = img.resize((224, 224))
    img = tf_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # normalize to [0,1] range
    return img

def classify_image(img):
    img_preprocessed = load_image(img)
    
    # Make predictions
    predictions = model.predict(img_preprocessed)
    predicted_class = np.argmax(predictions[0], axis=-1)
    predicted_label = imagenet_labels[predicted_class]
    confidence = tf.nn.softmax(predictions[0])[predicted_class]
    
    st.write(f'Predicted label: {predicted_label} with confidence: {confidence.numpy()*100:.2f}%')

def fetch_related_images(query):
    url = f"https://www.bing.com/images/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    image_elements = soup.find_all('a', {'class': 'iusc'})
    image_metas = [eval(img['m']) for img in image_elements if 'm' in img.attrs]
    images = [img_meta['murl'] for img_meta in image_metas]
    return images

def display_related_images(images):
    cols = st.columns(3)
    for i, image_url in enumerate(images[:3]):
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            cols[i].image(img, caption=image_url, width=200)
        except UnidentifiedImageError:
            st.error(f"Could not identify the image at URL: {image_url}")
            continue  # Skip to the next iteration of the loop

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Image Classifier</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', width=200)
    st.write("")
    st.write("Classifying...")
    classify_image(img)
    related_images = fetch_related_images(imagenet_labels[np.argmax(model.predict(load_image(img))[0], axis=-1)])
    st.write("Related Images:")
    display_related_images(related_images)
