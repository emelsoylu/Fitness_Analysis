import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import csv
from datetime import datetime
from scipy.signal import find_peaks
from collections import Counter
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

def predict_activity_from_video(video_path, model_path='knn_model.h5'):
    """
    Predict activity type from video using machine learning model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

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
    return Counter(predictions).most_common(1)[0][0] if predictions else "Unknown"


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate angle between three points
    """
    point_a, point_b, point_c = np.array(point_a), np.array(point_b), np.array(point_c)
    radians = np.arctan2(point_c[1]-point_b[1], point_c[0]-point_b[0]) - np.arctan2(point_a[1]-point_b[1], point_a[0]-point_b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def count_peaks(angle_list, prominence=10):
    """
    Count peaks in angle data to determine repetitions
    """
    peaks, _ = find_peaks(angle_list, prominence=prominence)
    return len(peaks)


def analyze_video_calorie(video_path, activity_class, weight):
    """
    Analyze video and calculate calories burned based on activity type and movement count
    """
    # Initialize angle tracking lists
    right_knee_angles = []
    left_knee_angles = []
    right_hip_angles = []
    left_hip_angles = []
    right_elbow_angles = []
    left_elbow_angles = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(frame_count / fps, 2)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Extract landmark coordinates
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angles
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Store angles
                right_knee_angles.append(right_knee_angle)
                left_knee_angles.append(left_knee_angle)
                right_hip_angles.append(right_hip_angle)
                left_hip_angles.append(left_hip_angle)
                right_elbow_angles.append(right_elbow_angle)
                left_elbow_angles.append(left_elbow_angle)

            except:
                continue

    cap.release()
    
    # Determine movement count and MET value based on activity type
    if activity_class == 'jumping' or activity_class == 'atlama':
        movement_count = count_peaks(left_knee_angles)
        # MET values based on jump intensity
        if movement_count < duration * 50 / 60:
            met_value = 5.65
        elif movement_count < duration * 100 / 60:
            met_value = 7.11
        else:
            met_value = 8.51
            
    elif activity_class == 'pullup' or activity_class == 'barfiks':
        movement_count = count_peaks(right_elbow_angles)
        # MET values based on pullup intensity
        if movement_count < duration * 30 / 60:
            met_value = 1.82
        elif movement_count < duration * 60 / 60:
            met_value = 2.43
        else:
            met_value = 5.47
            
    elif activity_class == 'cycling' or activity_class == 'bisiklet':
        movement_count = count_peaks(left_knee_angles)
        # MET values based on cycling intensity
        if movement_count < duration * 100 / 60:
            met_value = 5.83
        elif movement_count < duration * 200 / 60:
            met_value = 7.29
        else:
            met_value = 8.51
            
    elif activity_class == 'pushup':
        movement_count = count_peaks(right_elbow_angles)
        # MET values based on pushup intensity
        if movement_count < duration * 30 / 60:
            met_value = 1.82
        elif movement_count < duration * 45 / 60:
            met_value = 2.43
        else:
            met_value = 5.47
            
    elif activity_class == 'squat' or activity_class == 'squad':
        movement_count = count_peaks(right_knee_angles)
        # MET values based on squat intensity
        if movement_count < duration * 25 / 60:
            met_value = 5.5
        elif movement_count < duration * 35 / 60:
            met_value = 6.7
        else:
            met_value = 8.0
            
    elif activity_class == 'running' or activity_class == 'kosma':
        movement_count = count_peaks(left_hip_angles)
        # MET values based on running intensity
        if movement_count < duration * 100 / 60:
            met_value = 5.83
        elif movement_count < duration * 200 / 60:
            met_value = 7.29
        else:
            met_value = 11.55
    else:
        # Default values for unknown activities
        movement_count = count_peaks(right_hip_angles)
        met_value = 4.0

    # Calculate calories burned using MET formula
    # Calories = duration (hours) × MET × weight (kg)
    calorie = duration * met_value * weight / 3600  # Convert seconds to hours
    
    # Save results to CSV
    date = datetime.now().strftime("%Y-%m-%d")
    with open('calorie_results.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([date, video_path, duration, fps, activity_class, calorie, movement_count, weight])

    return {
        "date": date,
        "duration": duration,
        "activity": activity_class,
        "repetitions": movement_count,
        "intensity_MET": met_value,
        "calories_burned": calorie
    }


class FitnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fitness Analysis Application")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        self.setup_styles()
        
        self.weight = None
        self.video_path = None
        self.cap = None
        self.model = None
        self.is_playing = False
        self.current_frame = 0
        self.activity_type = None
        self.movement_count = 0
        
        self.create_widgets()
        self.load_model()
    
    def setup_styles(self):
        """
        Setup application styles and colors
        """
        self.colors = {
            'primary': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'purple': '#9C27B0',
            'background': '#f0f0f0',
            'white': '#ffffff',
            'text': '#333333'
        }
    
    def create_widgets(self):
        """
        Create GUI widgets with improved layout
        """
        # Main container with scrollbar
        self.main_canvas = tk.Canvas(self.root, bg=self.colors['background'])
        self.main_scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg=self.colors['background'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        # Pack canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to canvas
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Main content frame
        self.main_frame = tk.Frame(self.scrollable_frame, padx=20, pady=20, bg=self.colors['background'])
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="🏋️ Fitness Analysis Application", 
            font=('Arial', 18, 'bold'),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(0, 20))

        # Weight input frame
        self.weight_frame = tk.LabelFrame(
            self.main_frame, 
            text="👤 Weight Information", 
            font=('Arial', 12, 'bold'), 
            padx=15, pady=15,
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        self.weight_frame.pack(fill=tk.X, pady=(0, 10))
        
        weight_inner_frame = tk.Frame(self.weight_frame, bg=self.colors['white'])
        weight_inner_frame.pack(fill=tk.X)
        
        tk.Label(
            weight_inner_frame, 
            text="Your Weight (kg):", 
            font=('Arial', 12),
            bg=self.colors['white'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.weight_entry = tk.Entry(
            weight_inner_frame, 
            font=('Arial', 12), 
            width=12,
            relief=tk.RIDGE,
            bd=2
        )
        self.weight_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        self.submit_btn = tk.Button(
            weight_inner_frame, 
            text="💾 Save Weight", 
            command=self.save_weight, 
            font=('Arial', 11, 'bold'), 
            bg=self.colors['success'], 
            fg="white",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        )
        self.submit_btn.pack(side=tk.LEFT)

        # Video control frame
        self.video_frame = tk.LabelFrame(
            self.main_frame, 
            text="🎥 Video Controls", 
            font=('Arial', 12, 'bold'), 
            padx=15, pady=15,
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        self.video_frame.pack(fill=tk.X, pady=(0, 10))
        
        video_controls = tk.Frame(self.video_frame, bg=self.colors['white'])
        video_controls.pack(fill=tk.X)
        
        self.upload_btn = tk.Button(
            video_controls, 
            text="📁 Upload Video", 
            command=self.upload_video, 
            font=('Arial', 11, 'bold'), 
            state=tk.DISABLED, 
            bg=self.colors['primary'], 
            fg="white",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        )
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.play_btn = tk.Button(
            video_controls, 
            text="▶ Play", 
            command=self.play_video, 
            font=('Arial', 11, 'bold'), 
            state=tk.DISABLED, 
            bg=self.colors['warning'], 
            fg="white",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            video_controls, 
            text="⏹ Stop", 
            command=self.stop_video, 
            font=('Arial', 11, 'bold'), 
            state=tk.DISABLED, 
            bg=self.colors['danger'], 
            fg="white",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(
            video_controls, 
            text="🔍 Analyze Video", 
            command=self.analyze_video, 
            font=('Arial', 11, 'bold'), 
            state=tk.DISABLED, 
            bg=self.colors['purple'], 
            fg="white",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Video display frame
        self.video_display_frame = tk.LabelFrame(
            self.main_frame, 
            text="📺 Video Preview", 
            font=('Arial', 12, 'bold'),
            bg=self.colors['white'],
            fg=self.colors['text'],
            padx=10, pady=10
        )
        self.video_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_display = tk.Label(
            self.video_display_frame, 
            bg="black", 
            text="No video loaded",
            fg="white",
            font=('Arial', 14)
        )
        self.video_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress frame
        self.progress_frame = tk.LabelFrame(
            self.main_frame, 
            text="⏳ Progress", 
            font=('Arial', 12, 'bold'), 
            padx=15, pady=15,
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        self.progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(
            self.progress_frame, 
            variable=self.progress_var, 
            from_=0, to=100, 
            orient=tk.HORIZONTAL, 
            length=400, 
            showvalue=False,
            bg=self.colors['primary'],
            fg="white",
            troughcolor=self.colors['background']
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = tk.Label(
            self.progress_frame, 
            text="Ready to start", 
            font=('Arial', 11),
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        self.progress_label.pack()
        
        # Results frame with scrollbar
        self.result_frame = tk.LabelFrame(
            self.main_frame, 
            text="📊 Analysis Results", 
            font=('Arial', 12, 'bold'), 
            padx=15, pady=15,
            bg=self.colors['white'],
            fg=self.colors['text']
        )
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for text widget and scrollbar
        text_frame = tk.Frame(self.result_frame, bg=self.colors['white'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget with scrollbar
        self.result_text = tk.Text(
            text_frame, 
            height=12, 
            font=('Arial', 11), 
            wrap=tk.WORD,
            bg=self.colors['background'],
            fg=self.colors['text'],
            relief=tk.SUNKEN,
            bd=2,
            padx=10,
            pady=10
        )
        
        result_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert welcome message
        welcome_msg = """Welcome to Fitness Analysis Application! 🎉

📋 Instructions:
1. Enter your weight in kg and click 'Save Weight'
2. Upload a video file of your exercise
3. Use video controls to preview your video
4. Click 'Analyze Video' to get detailed results

📈 The application will analyze your exercise and provide:
• Activity type recognition
• Movement repetition counting
• Calorie burn calculation
• Exercise intensity measurement

Ready to start your fitness journey? 💪"""
        
        self.result_text.insert(tk.END, welcome_msg)
        self.result_text.config(state=tk.DISABLED)
    
    def _on_mousewheel(self, event):
        """
        Handle mouse wheel scrolling
        """
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def load_model(self):
        """
        Load machine learning model for activity classification
        """
        try:
            model_path = 'knn_model.h5'
            if not os.path.exists(model_path):
                messagebox.showwarning("Warning", f"Model file not found: {model_path}")
                return
            self.model = joblib.load(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def save_weight(self):
        """
        Save user weight input
        """
        try:
            weight_text = self.weight_entry.get().strip()
            if not weight_text:
                raise ValueError("Weight value cannot be empty")
            self.weight = float(weight_text)
            if self.weight <= 0:
                raise ValueError("Weight must be positive")
            self.upload_btn.config(state=tk.NORMAL)
            self.weight_entry.config(state=tk.DISABLED)
            self.submit_btn.config(state=tk.DISABLED)
            messagebox.showinfo("Success", "Weight saved!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.weight_entry.delete(0, tk.END)

    def upload_video(self):
        """
        Upload video file
        """
        file_path = filedialog.askopenfilename(
            title="Select Video File", 
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All Files", "*.*")]
        )
        if file_path and os.path.exists(file_path):
            if self.cap:
                self.cap.release()
            self.video_path = os.path.abspath(file_path)
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video!")
                return
            self.play_btn.config(state=tk.NORMAL)
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.show_frame()

    def show_frame(self, frame=None):
        """
        Display video frame in GUI
        """
        if self.cap is None:
            return
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        ratio = min(800/w, 600/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        frame = cv2.resize(frame, (new_w, new_h))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_display.imgtk = imgtk
        self.video_display.configure(image=imgtk)

    def play_video(self):
        """
        Play or pause video
        """
        if not self.is_playing:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            self.stop_btn.config(state=tk.NORMAL)
            self.play_video_frames()
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")

    def play_video_frames(self):
        """
        Play video frames continuously
        """
        if self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    progress = (self.current_frame / total_frames) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"Playing: {progress:.1f}%")
                self.show_frame(frame)
                self.root.after(30, self.play_video_frames)
            else:
                self.stop_video()

    def stop_video(self):
        """
        Stop video playback
        """
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.progress_var.set(0)
            self.progress_label.config(text="Ready")
            self.show_frame()

    def classify_activity(self, cap):
        """
        Classify activity type using machine learning model
        """
        try:
            return predict_activity_from_video(self.video_path, model_path='knn_model.h5')
        except Exception as e:
            print(f"Error in classification: {e}")
            return "General Exercise"

    def analyze_video(self):
        """
        Analyze video and calculate calories
        """
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded!")
            return
        if not self.weight:
            messagebox.showerror("Error", "Weight is required!")
            return
            
        self.stop_video()
        self.analyze_btn.config(state=tk.DISABLED, text="🔄 Analyzing...")
        self.progress_label.config(text="Analyzing video... Please wait")
        
        # Enable text widget for updates
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🔄 Analysis in progress...\n\n")
        self.result_text.update()
        
        try:
            # Classify activity type
            self.result_text.insert(tk.END, "🎯 Classifying activity type...\n")
            self.result_text.update()
            
            cap = cv2.VideoCapture(self.video_path)
            self.activity_type = self.classify_activity(cap)
            cap.release()
            
            self.result_text.insert(tk.END, f"✅ Activity detected: {self.activity_type}\n\n")
            self.result_text.insert(tk.END, "📊 Calculating calories and movement analysis...\n")
            self.result_text.update()
            
            # Calculate calories using integrated function
            analysis_result = analyze_video_calorie(self.video_path, self.activity_type, self.weight)
            
            # Clear and display final results
            self.result_text.delete(1.0, tk.END)
            
            # Create formatted result display
            result_text = f"""🎉 ANALYSIS COMPLETE! 🎉

📁 FILE INFORMATION:
   • Video: {os.path.basename(self.video_path)}
   • Duration: {analysis_result['duration']:.2f} seconds ({analysis_result['duration']/60:.1f} minutes)
   • Analysis Date: {analysis_result['date']}
   • Analysis Time: {datetime.now().strftime('%H:%M:%S')}

🏃 EXERCISE DETAILS:
   • Activity Type: {analysis_result['activity'].title()}
   • Repetitions Counted: {analysis_result['repetitions']}
   • Exercise Intensity (MET): {analysis_result['intensity_MET']:.2f}

🔥 CALORIE INFORMATION:
   • Total Calories Burned: {analysis_result['calories_burned']:.2f} kcal
   • Your Weight: {self.weight} kg
   • Calories per Minute: {(analysis_result['calories_burned'] / (analysis_result['duration']/60)):.2f} kcal/min

📈 PERFORMANCE METRICS:
   • Repetitions per Minute: {(analysis_result['repetitions'] / (analysis_result['duration']/60)):.1f} reps/min
   • Workout Intensity: {'High' if analysis_result['intensity_MET'] > 6 else 'Medium' if analysis_result['intensity_MET'] > 4 else 'Low'}

💡 RECOMMENDATIONS:
   • {'Great job! High intensity workout!' if analysis_result['intensity_MET'] > 6 else 'Good workout! Try to increase intensity for more calories.' if analysis_result['intensity_MET'] > 4 else 'Light workout. Consider increasing pace or duration.'}
   • Keep up the great work! 💪

📝 Note: Results have been saved to 'calorie_results.csv' for tracking your progress."""
            
            self.result_text.insert(tk.END, result_text)
            self.progress_label.config(text="✅ Analysis complete! Check results below.")
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"❌ ANALYSIS FAILED\n\nError: {str(e)}\n\nPlease check your video file and try again.")
            self.progress_label.config(text="❌ Analysis failed!")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
        finally:
            self.analyze_btn.config(state=tk.NORMAL, text="🔍 Analyze Video")
            self.result_text.config(state=tk.DISABLED)  # Disable editing


if __name__ == "__main__":
    root = tk.Tk()
    app = FitnessApp(root)
    root.mainloop()