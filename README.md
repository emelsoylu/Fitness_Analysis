# Fitness_Analysis
Fitness Analysis Software


# predict_class.py
This console-based video activity recognition system analyzes human movements in video files to identify specific activities. Using MediaPipe's advanced pose estimation technology, the program extracts body landmark coordinates from 10 evenly sampled frames of the input video. These spatial features are then processed by a pre-trained KNN classifier (loaded from 'knn_model.h5') which predicts the most likely activity. The system accepts common video formats (MP4, AVI, MOV) through simple command-line prompts, validates the input file, and displays the final prediction after processing. Designed for ease of use, it handles the entire workflow from video input to activity classification without requiring complex setup, making it suitable for basic human activity recognition tasks in offline environments. The implementation focuses on core functionality while maintaining clear error messages and user guidance throughout the console interface.
