import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
import chromadb
import sqlite3
import json
import uuid
import os
from datetime import datetime
import time

MODEL_ROOT = '/home/invisa/Desktop/my_grad_streamlit/insightface_model'
MODEL_NAME = 'antelopev2'







DETECTION_SIZE = (640, 640)
RTSP_URL = "rtsp://admin:Admin%40123@192.168.1.64:554/Streaming/Channels/101"
CHROMA_STORE_PATH = "./store"
DATABASE_PATH = 'attendance_system.db'

# Initialize face analysis model
try:
    app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
except Exception as e:
    st.error(f"Failed to initialize face analysis model: {str(e)}")
    st.stop()

def create_or_add_to_collection(collection_name, path_to_chroma="./store"):
    """
    Create a new collection or add data to an existing one in ChromaDB.

    Args:
        collection_name (str): The name of the collection.
        path_to_chroma (str): Path to the ChromaDB store.

    Returns:
        collection: The ChromaDB collection object.
    """
    try:
        # Initialize the ChromaDB client with the persist directory
        client = chromadb.PersistentClient(path_to_chroma)

        # Check if the collection already exists
        collection_names = client.list_collections()
        if collection_name in collection_names:
            collection = client.get_collection(name=collection_name)
        else:
            # Create a new collection if it doesn't exist
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # Retrieve data from the SQLite database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, facial_features FROM presidents_embeds")
        rows = cursor.fetchall()
        conn.close()

        # Add data to the collection
        for index, row in enumerate(rows):
            name, embedding_str = row
            embedding = json.loads(embedding_str)
            collection.add(
                ids=[str(uuid.uuid4())],  # Generate a unique ID for each entry
                documents=[name],
                embeddings=[embedding],
                metadatas=[{"name": name}],
            )

        print("Data successfully stored in ChromaDB!")
        return collection

    except Exception as e:
        st.error(f"Error creating or adding to the collection: {str(e)}")
        return None

def log_attendance(name):
    """Log attendance in the SQLite database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if the person already exists in the attendance log
        cursor.execute("SELECT COUNT(*) FROM attendance_log WHERE name = ?", (name,))
        count = cursor.fetchone()[0]
        if count > 0:
            return  # Do not log again if the name already exists
        
        # Log the attendance
        cursor.execute("INSERT INTO attendance_log (name, timestamp) VALUES (?, ?)", (name, datetime.now()))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error logging attendance: {str(e)}")

def process_frame(frame, threshold=0.6, collection=None):
    """Process a frame with face recognition"""
    faces = app.get(frame)
    
    if not faces:
        return frame
    
    for face in faces:
        x1, y1, x2, y2 = face['bbox'].astype(int)
        embedding = face['embedding']
        
        # Get recognition results
        name, confidence = cosine_similarity_search(embedding, threshold, collection)
        
        # Log attendance if a person is recognized
        if name != "Unknown":
            log_attendance(name)
        # Draw results
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
        if label == 'Unknown':
            label = ''
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def cosine_similarity_search(query_embedding, threshold=0.6, collection=None):
    """Enhanced cosine similarity search with debugging"""
    if collection is None:
        return "Unknown", 0.0

    # Normalize query embedding
    query_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    results = collection.query(
        query_embeddings=[query_normalized.tolist()],
        n_results=1,
        include=["distances", "metadatas"]
    )
    
    if not results['ids'][0]:
        return "Unknown", 0.0
    
    # Convert distance to similarity
    similarity = 1 - results['distances'][0][0]
    
    # Debug print
    print(f"Best match: {results['metadatas'][0][0]['name']} (Similarity: {similarity:.2f})")
    
    return (results['metadatas'][0][0]['name'], similarity) if similarity >= threshold else ("Unknown", similarity)

def show_real_time_prediction():
    # Load embeddings to ChromaDB
    collection = create_or_add_to_collection("face_recognition", path_to_chroma=CHROMA_STORE_PATH)
    
    st.header("Real-Time Face Recognition")
    threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.6)

    # Video capture with hic camera
    cap = cv2.VideoCapture(RTSP_URL)
    # laptop camera
    # cap = cv2.VideoCapture(0)
    # video_path = '/home/invisa/Desktop/my_grad_streamlit/sisi.mp4'
    # cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        st.error("Error: Could not open video stream. Please check the camera connection.")
        return  # Exit the function if the camera connection fails

    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from stream. Please check the camera connection.")
            break
        
        try:
            # time.sleep(5)
            processed_frame = process_frame(frame, threshold, collection)
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            continue
        
        # Convert frame to RGB
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        stframe.image(processed_frame, channels="RGB", use_container_width=True)

    cap.release()


if __name__ == "__main__":
    show_real_time_prediction()
