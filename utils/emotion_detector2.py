# app/utils/emotion_detector.py

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import os

# --- Configuration (Paths are relative to the project root, assuming 'streamlit run app/app.py' is executed from root) ---
FACE_CASCADE_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'haarcascade_files/haarcascade_eye.xml'
EMOTION_MODEL_PATH = 'models/model_3.keras' # Path to your trained model in the 'models' folder

EMOTIONS = ['happy', 'neutral', 'sad', 'surprise'] 
IMG_SIZE = 48 

class EmotionDetector:
    """
    Encapsulates face detection, eye detection (attention),
    and emotion prediction using the trained CNN model.
    """
    
    def __init__(self):
        # 1. Load Cascade Classifiers
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        if self.face_cascade.empty() or self.eye_cascade.empty():
            # A friendly way to notify the user if files are missing
            raise FileNotFoundError("FATAL ERROR: Failed to load Haar Cascades. Check that 'haarcascade_files/' exists in the root.")
        
        # 2. Load Keras Model
        try:
            # Check if model file exists before loading
            if not os.path.exists(EMOTION_MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {EMOTION_MODEL_PATH}")
                
            self.emotion_classifier = load_model(EMOTION_MODEL_PATH)
        except Exception as e:
            raise Exception(f"FATAL ERROR: Failed to load Keras model from {EMOTION_MODEL_PATH}. Error: {e}")
            
    def process_frame(self, frame):
        """
        Processes a single video frame for analysis.
        
        Returns:
            tuple: (annotated_frame, emotion_canvas, attention_status, emotion_label)
        """
        # 1. Pre-processing
        # Handle cases where frame might be None (e.g., during video stream initialization)
        if frame is None:
            return None, np.zeros((350, 400, 3), dtype="uint8"), False, 'No Frame'

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Initialize outputs
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
        emotion_canvas = np.zeros((350, 400, 3), dtype="uint8")
        attention_status = False
        emotion_label = 'No Face'
        
        if len(faces) == 0:
            cv2.putText(frame, "STATUS: Not-Attentive (No Face)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame, emotion_canvas, False, 'No Face'
            
        # Process the largest face found
        (x, y, w, h) = faces[0] 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2) # Orange rectangle
        
        roi_gray_input = gray[y:y+h, x:x+w]
        
        # 2. Eye Detection (Attention)
        eyes = self.eye_cascade.detectMultiScale(roi_gray_input)
        
        if len(eyes) >= 1:
            attention_status = True
            for (ex, ey, ew, eh) in eyes[:2]:
                # Draw green eye rectangles on the original frame ROI
                cv2.rectangle(frame[y:y+h, x:x+w], (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

        # 3. Emotion Prediction Input Preparation
        if roi_gray_input.size == 0:
            return frame, emotion_canvas, attention_status, 'Error ROI'

        roi_input = cv2.resize(roi_gray_input, (IMG_SIZE, IMG_SIZE))
        roi_input = roi_input.astype("float") / 255.0
        roi_input = img_to_array(roi_input)
        roi_input = np.expand_dims(roi_input, axis=0) 

        # 4. Prediction
        # Ensure predict runs silently
        preds = self.emotion_classifier.predict(roi_input, verbose=0)[0] 
        emotion_label = EMOTIONS[preds.argmax()]

        # 5. Draw Probability Bars on the canvas
        canvas_bkg = (30, 30, 30) # Dark gray background
        cv2.rectangle(emotion_canvas, (0, 0), (400, 350), canvas_bkg, -1) 
        
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.1f}%".format(emotion.upper(), prob * 100) 
            w_bar = int(prob * 350)
            
            # Color coding for emotions
            if emotion == 'happy': bar_color = (0, 255, 0)
            elif emotion == 'neutral': bar_color = (255, 255, 0) 
            else: bar_color = (0, 0, 255) # Red/Blue for sad/surprise
            
            # Draw probability bar
            cv2.rectangle(emotion_canvas, (20, (i * 50) + 5), (w_bar + 20, (i * 50) + 40), bar_color, -1)
            
            # Draw text
            cv2.putText(emotion_canvas, text, (25, (i * 50) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 6. Draw Attention Status on Frame
        att_status_text = 'Attentive' if attention_status else 'Distracted'
        att_color = (0, 255, 0) if attention_status else (0, 0, 255)
        
        label_text = f"{att_status_text} ({emotion_label.upper()})"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, att_color, 2)
        cv2.putText(frame, f"Emotion: {emotion_label.upper()}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        return frame, emotion_canvas, attention_status, emotion_label