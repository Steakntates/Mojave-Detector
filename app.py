import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Mojave Detector", page_icon="🧭")

st.title("🧭 Mojave Desert Quick Detector")
st.markdown("Detect **Outcroppings** or **Alteration Zones** from satellite images.")

st.sidebar.header("Detection Settings")
detection_type = st.sidebar.selectbox("Choose Detection Type", ("Outcrops", "Alteration Zones"))
sensitivity = st.sidebar.slider("Sensitivity", 0, 255, 100)

uploaded_file = st.file_uploader("Upload a Satellite Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)

    st.subheader("Detection Results")

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    if detection_type == "Outcrops":
        _, mask = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)
        st.image(mask, caption="Outcrop Detection", use_column_width=True)

    elif detection_type == "Alteration Zones":
        red_channel = img_np[:, :, 0]
        _, mask = cv2.threshold(red_channel, sensitivity, 255, cv2.THRESH_BINARY)
        st.image(mask, caption="Alteration Zone Detection", use_column_width=True)

    st.download_button(
        "Download Detection Mask",
        data=Image.fromarray(mask).tobytes(),
        file_name="detection_mask.png",
        mime="image/png"
    )

else:
    st.info("📂 Upload an image to start.")
import streamlit as st
import leafmap.foliumap as leafmap

st.title("🛰️ Mojave Desert Live Map")

# Create a map centered on the Mojave Desert
m = leafmap.Map(center=(35.0, -115.5), zoom=7)

# Add a basemap
m.add_basemap("SATELLITE")

# Display the map inside Streamlit
m.to_streamlit(width=800, height=600)
