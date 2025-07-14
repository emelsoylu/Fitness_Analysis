import os
import cv2
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import mediapipe as mp

def predict_activity_from_video(video_path, model_path='knn_model.h5'):
    """Predict activity from video using pre-trained KNN model"""
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = [int(i * frame_count / 10) for i in range(10)]
    predictions = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                row = [coord for lm in pose for coord in (lm.x, lm.y, lm.z, lm.visibility)]
                X = pd.DataFrame([row])
                pred = model.predict(X)[0]
                predictions.append(pred)

    cap.release()

    if predictions:
        return Counter(predictions).most_common(1)[0][0]
    else:
        return "Unknown"

def main():
    print("=== Video Activity Recognition ===")
    print("Please enter the path to your video file (e.g., C:/videos/activity.mp4)")
    
    while True:
        video_path = input("Video file path: ").strip()
        
        if not os.path.exists(video_path):
            print("Error: File not found. Please try again.")
            continue
        
        if not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            print("Error: Unsupported file format. Please use MP4, AVI, or MOV.")
            continue
            
        try:
            print("\nProcessing video... Please wait...")
            predicted_class = predict_activity_from_video(video_path)
            print(f"\nPredicted activity: {predicted_class}")
            break
        except Exception as e:
            print(f"\nError processing video: {str(e)}")
            print("Please try again with a different video file.\n")

if __name__ == "__main__":
    main()
    print("\nPress Enter to exit...")
    input()