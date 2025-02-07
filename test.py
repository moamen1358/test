import streamlit as st
import cv2
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the RTSP stream URL
stream_url = "rtsp://admin:Admin%40123@192.168.1.64:554/Streaming/Channels/101"

# Streamlit title for the app
st.title("RTSP Stream Viewer")

# Function to connect to the RTSP stream with retries
def connect_to_stream(url, retries=5, timeout=60000):
    for i in range(retries):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Increase Open and Read timeout
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout)
        
        if cap.isOpened():
            logging.info("Successfully connected to the stream.")
            return cap
        else:
            logging.warning(f"Retrying to connect... ({i+1}/{retries})")
            time.sleep(2)
    return None

# Try to connect to the stream
cap = connect_to_stream(stream_url)

if not cap:
    st.error("Failed to connect to the stream after retries.")
else:
    # Streamlit placeholder for displaying video frames
    frame_placeholder = st.empty()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            logging.error("Failed to grab frame. Stopping stream.")
            st.error("Failed to grab frame from stream.")
            break

        # Convert the frame from BGR (OpenCV default) to RGB (Streamlit uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Stop the loop if needed (implement any condition or just continue indefinitely)
        # You can add a Streamlit button to stop the stream if desired.
