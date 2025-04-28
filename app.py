import streamlit as st
import numpy as np
import cv2
from PIL import Image
import leafmap.foliumap as leafmap

st.set_page_config(page_title="Mojave Detector", page_icon="🧭", layout="wide")

# --- App Title ---
st.title("🧭 Mojave Desert Feature Detection App")

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Detection Tool", "🌎 Live Mojave Map"])

# --- Tab 1: Detection ---
with tab1:
    st.header("🔍 Detect Outcroppings or Alteration Zones")

    detection_type = st.sidebar.selectbox("Choose Detection Type", ("Outcrops", "Alteration Zones"))
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0, 255, 100)

    uploaded_file = st.file_uploader("Upload a Satellite Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        st.subheader("Detection Results")

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
        st.info("📂 Upload a satellite image to begin detection.")

# --- Tab 2: Live Map ---
with tab2:
    st.header("🌎 Live Satellite Map - Mojave Desert")

    m = leafmap.Map(center=(35.0, -115.5), zoom=7)
    m.add_basemap("SATELLITE")
    m.to_streamlit(height=700)
