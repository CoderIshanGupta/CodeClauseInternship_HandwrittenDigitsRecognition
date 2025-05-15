import streamlit as st
import numpy as np
import cv2
import torch
from model import DeepCNN
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Page setup
st.set_page_config(page_title="Digit Recognizer", layout="wide", page_icon="✍️")

# Load the trained model
model = DeepCNN()
model.load_state_dict(torch.load("deep_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Sidebar Info
with st.sidebar:
    st.title("📘 About")
    st.markdown("""
        This app allows you to **draw or upload** a handwritten digit.

        ➤ Uses a custom-trained CNN model  
        ➤ Recognizes digits 0 through 9  
        ➤ Try different styles to test accuracy  

        Built with ❤️ using Python + Streamlit.
    """)
    st.caption("💻 Created by: Ishan Gupta")

# App title
st.title("✍️ Handwritten Digit Recognition")

# Choose input method
mode = st.radio("Choose input method:", ["Draw Digit", "Upload Image"], horizontal=True)

# Preprocessing function
def preprocess_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.resize(image, (28, 28))
    image = np.invert(image)
    image = image / 255.0
    image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
    return image

# Drawing mode
if mode == "Draw Digit":
    col1, col2 = st.columns([1, 1])
    with col1:
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )

    with col2:
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

            if np.sum(gray < 250) > 100:
                input_tensor = preprocess_image(gray)
                output = model(input_tensor)
                prediction = torch.argmax(output).item()

                st.image(gray, caption="Drawn Digit", width=280, clamp=True, channels="GRAY")
                st.subheader(f"Prediction: **{prediction}**")
            else:
                st.warning("🖌️ Canvas appears to be empty. Please draw a digit.")

# Upload mode
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        input_tensor = preprocess_image(img_array)

        output = model(input_tensor)
        prediction = torch.argmax(output).item()

        st.image(img, caption="Uploaded Digit", use_container_width=True)
        st.subheader(f"Prediction: **{prediction}**")
