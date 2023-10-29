import streamlit as st, requests, tensorflow as tf, tensorflow_hub as hub
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from tensorflow.keras.preprocessing import image as tf_img
import numpy as np

fe_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(fe_url, input_shape=(224, 224, 3))])
labels = np.array(open(tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')).read().splitlines())

def process_image(img):
    img = img.resize((224, 224))
    img = tf_img.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

def classify_image(img):
    img_prep = process_image(img)
    preds = model.predict(img_prep)
    pred_class = np.argmax(preds[0], axis=-1)
    pred_label = labels[pred_class]
    confidence = tf.nn.softmax(preds[0])[pred_class]
    st.write(f'Predicted label: {pred_label} with confidence: {confidence.numpy()*100:.2f}%')

def fetch_and_display_related_images(query):
    url = f"https://www.bing.com/images/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    img_elements = soup.find_all('a', {'class': 'iusc'})
    img_metas = [eval(img['m']) for img in img_elements if 'm' in img.attrs]
    images = [img_meta['murl'] for img_meta in img_metas]

    cols = st.columns(3)
    for i, img_url in enumerate(images[:3]):
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            cols[i].image(img, caption=img_url, width=200)
        except UnidentifiedImageError:
            st.error(f"Could not identify the image at URL: {img_url}")
            continue

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Image Classifier</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', width=200)
    st.write("")
    st.write("Classifying...")
    classify_image(img)
    query = labels[np.argmax(model.predict(process_image(img))[0], axis=-1)]
    st.write("Related Images:")
    fetch_and_display_related_images(query)
