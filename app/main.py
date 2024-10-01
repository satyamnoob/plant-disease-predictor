import os
import json
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

# Path configurations
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Loading the class name
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to convert image to greyscale and save
def convert_to_greyscale(image_path, save_path=None):
    img = cv2.imread(image_path)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if save_path:
        cv2.imwrite(save_path, grey_img)
    return grey_img

# Function to segment image and save
def segment_image(img):
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    # Create a mask for the green color
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Define the range for brown/black color in HSV (for disease spots)
    lower_brown_black = np.array([0, 0, 0])
    upper_brown_black = np.array([180, 255, 50])

    # Create a mask for the brown/black color
    disease_mask = cv2.inRange(hsv_img, lower_brown_black, upper_brown_black)

    # Combine the masks to retain both leaf and disease spots
    combined_mask = cv2.bitwise_or(green_mask, disease_mask)

    # Use morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    # Smooth the edges of the mask
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Create a 3-channel mask to apply on the original image
    mask_3_channel = cv2.merge([combined_mask, combined_mask, combined_mask])

    # Segment the leaf by masking the original image
    segmented_img = cv2.bitwise_and(img, mask_3_channel)

    # Set the background to black
    background = np.ones_like(img, np.uint8) * 0
    background_mask = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(combined_mask))
    segmented_img = cv2.add(segmented_img, background_mask)

    return segmented_img

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.convert('RGB')
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV format (BGR)
    
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Uploaded Image')

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

            # Segment the uploaded image
            segmented_img = segment_image(img_array)

            # Convert segmented image from BGR to RGB format for displaying
            segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)

            # Display the segmented image
            st.image(segmented_img_rgb, caption='Segmented Image', use_column_width=True)
