import streamlit as st  # type: ignore
from keras.models import load_model  # type: ignore
from PIL import Image, ImageOps  # type: ignore
import numpy as np  # type: ignore
from streamlit_option_menu import option_menu  # type: ignore
import json  # To read the JSON file

# Load the model once globally
@st.cache_resource  # Cache the model to avoid reloading it for each prediction
def load_keras_model():
    model = load_model("keras_model.h5", compile=False)
    return model

# Load labels once globally
@st.cache_resource
def load_labels():
    with open("labels.txt", "r") as file:
        class_names = file.readlines()
    return class_names

# TensorFlow Model Prediction Function
def model_prediction(test_image):
    model = load_keras_model()
    class_names = load_labels()

    # Create the array for the image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open image and process
    image = Image.open(test_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array and normalize it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove any leading/trailing whitespace
    confidence_score = prediction[0][index]

    return class_name, confidence_score  # Return the class and confidence

# Function to clean the predicted class name
def clean_disease_name(class_name):
    return class_name.split(" ", 1)[1].strip()  # Removes the number and any leading spaces

# ----------Navigation Bar (Horizontal)----------#
selected = option_menu(
    menu_title=None,  # No title
    options=["Home", "Predict", "About", "How It Works", "Recommendations"],  # Menu options
    icons=["house", "graph-up", "info", "lightbulb", "clipboard-check"],  # Icons for the options
    menu_icon="cast",  # Icon for the menu button
    default_index=0,  # Default active index (Home)
    orientation="horizontal",  # Horizontal layout
)

# ----------Page Logic----------#
if selected == "Home":
    st.markdown("""
    <h1 style='text-align: center; padding-bottom: 25px;'>Lumpy Skin Disease Detection</h1>
    """, unsafe_allow_html=True)

    # Columns for layout
    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.image("cow.png", width=300)

    with col2:
        st.markdown("""
            <div style='text-align: justify;'>
            Welcome to the Lumpy Skin Disease Detection System! This web-based application is designed to assist livestock owners and veterinarians in identifying Lumpy Skin Disease in cows. By uploading an image of the affected area, the system leverages advanced machine learning techniques to analyze the image and provide a quick and accurate diagnosis. 
            The aim is to empower users with actionable insights to protect animal health and ensure effective disease management.
            </div>
        """, unsafe_allow_html=True)

elif selected == "Predict":
    st.header("Lumpy Skin Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, use_container_width=True)

        # Predict button
        if st.button("Predict"):
            st.write("Analyzing the image...")

            # Get the prediction and confidence
            class_name, confidence_score = model_prediction(test_image)

            # Clean the class name by removing numbers and spaces
            cleaned_class_name = clean_disease_name(class_name)

            # Display results
            st.success(f"Model Prediction: {cleaned_class_name}")
            st.info(f"Confidence Score: {confidence_score:.2f}")

elif selected == "About":
    st.header("About")
    st.markdown("""### About the Project
This application is a **computer science project** submitted to **Great Zimbabwe University** by **Tonderayi Selestino Runyowa (M204059)**.

### Dataset Source
The dataset used for training this model is available at [Kaggle](https://www.kaggle.com/datasets/amankukar/lumpy-skin).

### Labels
- **0 Lumpy Skin:** Indicates the presence of Lumpy Skin Disease.
- **1 Normal Skin:** Indicates healthy skin.

This system aims to provide accurate predictions and confidence scores to assist users in timely decision-making.
""")

elif selected == "How It Works":
    st.header("How It Works")
    st.markdown("""### How It Works
1. **Upload Image:** Go to the **Lumpy Skin Disease Recognition** page and upload an image of the cow's skin.
2. **Analysis:** The system processes the image using a trained deep learning model to identify signs of Lumpy Skin Disease.
3. **Results:** The predicted class and confidence score are displayed for user interpretation.

### Why Use This App?
- **Accurate:** Built with state-of-the-art machine learning algorithms.
- **User-Friendly:** Intuitive interface for seamless interaction.
- **Fast:** Results provided within seconds.

### Get Started
Click on the **Predict** page to upload an image and receive a diagnosis!
""")

elif selected == "Recommendations":
    st.header("Recommendations")
    st.markdown("""### Precautions to Prevent and Heal Lumpy Skin Disease
- **Vaccination:** Ensure all livestock are vaccinated against Lumpy Skin Disease.
- **Hygiene:** Maintain clean and sanitary conditions in cattle housing to reduce the risk of infection.
- **Insect Control:** Use insecticides to control vectors like flies and mosquitoes that can spread the disease.
- **Isolation:** Isolate infected animals immediately to prevent the spread to healthy livestock.
- **Nutrition:** Provide a balanced diet to boost the immunity of affected and healthy animals.
- **Consultation:** Regularly consult veterinarians for disease management and early diagnosis.

By following these steps, the risk of infection can be minimized, and recovery can be accelerated.
""")
