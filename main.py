import streamlit as st
from model_helper import predict

st.title("Clothing Classification")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:  # Check if a file has been uploaded
    image_path = "temp_file.jpg" # Define a temporary file path to save the uploaded image
    with open(image_path, "wb") as f: # Open the temporary file in write-binary mode
        f.write(uploaded_file.getbuffer()) # Write the uploaded image's data to the temporary file
        st.image(uploaded_file, caption="Uploaded File") # Display the uploaded image in the Streamlit app
        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}") # Display the prediction in an info box in the Streamlit app
