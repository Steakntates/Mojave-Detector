import streamlit as st
import numpy as np
import cv2
from PIL import Image
import leafmap.foliumap as leafmap
import tempfile
import os

st.set_page_config(page_title="Mojave Detector", page_icon="🧭", layout="wide")

# --- App Title ---
st.title("🧭 Mojave Desert Feature Detection App")

# --- Session state to share data between tabs ---
if "mask_file" not in st.session_state:
    st.session_state["mask_file"] = None
if "bounds" not in st.session_state:
    st.session_state["bounds"] = None
if "map_center" not in st.session_state:
    st.session_state["map_center"] = (35.0, -115.5)
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 7

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

        # Save mask temporarily as PNG with transparency
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = 255  # Red
        mask_rgba[:, :, 3] = mask  # Alpha channel (transparency)

        temp_dir = tempfile.mkdtemp()
        mask_path = os.path.join(temp_dir, "mask_overlay.png")
        Image.fromarray(mask_rgba).save(mask_path)

        # Save path to session state
        st.session_state["mask_file"] = mask_path

        # Dynamically generate bounds based on image size
        center_lat, center_lon = 35.0, -115.5  # Center of Mojave Desert
        meters_per_pixel = 1.0  # Assume 1 meter per pixel
        degrees_per_meter = 1 / 111320  # Rough

        height, width = mask.shape
        delta_lat = (height * meters_per_pixel) * degrees_per_meter
        delta_lon = (width * meters_per_pixel) * degrees_per_meter

        bounds = [
            (center_lat - delta_lat / 2, center_lon - delta_lon / 2),  # SouthWest
            (center_lat + delta_lat / 2, center_lon + delta_lon / 2),  # NorthEast
        ]
        st.session_state["bounds"] = bounds

        # Set map center and zoom dynamically
        st.session_state["map_center"] = (center_lat, center_lon)

        # Calculate zoom level based on width
        if width <= 500:
            st.session_state["map_zoom"] = 17
        elif width <= 1000:
            st.session_state["map_zoom"] = 16
        elif width <= 3000:
            st.session_state["map_zoom"] = 15
        else:
            st.session_state["map_zoom"] = 13

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

    m = leafmap.Map(center=st.session_state["map_center"], zoom=st.session_state["map_zoom"])
    m.add_basemap("SATELLITE")

    # If there is a detection mask, add it as an overlay
    if st.session_state["mask_file"] and st.session_state["bounds"]:
        m.add_image(
            st.session_state["mask_file"],
            bounds=st.session_state["bounds"],
            opacity=0.5,
            name="Detection Overlay",
        )

    m.to_streamlit(height=700)
