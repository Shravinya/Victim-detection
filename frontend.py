import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests

# Load the victim detection model
model = tf.keras.models.load_model('victim_detection_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Function to predict whether victim is present or not
def predict_victim(img, model):  # Pass model as an argument
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    prediction_label = "Victim Detected" if prediction[0] > 0.5 else "No Victim Detected"
    return prediction_label, prediction[0]

# Function to send prediction result to backend
def send_prediction_result(prediction_result, backend_url):
    data = {"prediction_result": prediction_result}
    response = requests.post(backend_url, json=data)
    if response.status_code == 200:
        st.success("Prediction result sent successfully to the backend.")
    else:
        st.error("Failed to send prediction result to the backend.")

# Streamlit App
def main():
    st.title("Victim Detection ")
    st.markdown("---")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    if uploaded_file is not None:
        st.markdown("---")
        # Display the uploaded imagesend.pytreamlit run front

        st.subheader("Uploaded Image:")
        image_display = image.load_img(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)

        # Make prediction when the 'Predict' button is clicked
        if st.button("Predict", key="predict_button"):
            st.markdown("---")
            # Pass model to predict_victim function
            prediction_result, confidence = predict_victim(image_display, model)
            st.subheader("Prediction Result:")
            if prediction_result == "Victim Detected":
                st.success(f"**Prediction:** {prediction_result}")
            else:
                st.error(f"**Prediction:** {prediction_result}")
            st.write("**Confidence:** {:.2%}".format(float(confidence)))  # Convert confidence to float

            # Specify your backend URL
            backend_url = "http://127.0.0.1:5000/"  # Replace this with your backend URLhttp://127.0.0.1:5000/streamlit run frontend.py

            # Send prediction result to backend
            send_prediction_result(prediction_result, backend_url)

if __name__ == "__main__":
    main()
