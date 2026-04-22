from keras.preprocessing.image import img_to_array
import cv2
import imutils
from keras.models import load_model
import numpy as np
import os

# --- Configuration ---
# NOTE: Ensure these paths are correct relative to where you run the script!
FACE_CASCADE_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'haarcascade_files/haarcascade_eye.xml'

# 1. UPDATED MODEL PATH: Must match where your Grayscale Model 3 was saved.
video_emotion_model_path = './model_3.keras' 

video_file_name = "sample/test2.mp4"
use_live_video=False # Set to True to use your webcam

# 2. CRITICAL CHANGE: Updated to match the 4 classes of model_3.
# The order must match the order used during training (e.g., in flow_from_directory).
EMOTIONS = ['happy', 'neutral', 'sad', 'surprise'] 

# --- Initialize Cascade Classifiers ---
# We assume the cascade file names/paths have been fixed from previous errors
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    print(f"FATAL ERROR: Failed to load Face Cascade. Check path: {FACE_CASCADE_PATH}")
    exit()

if eye_cascade.empty():
    print(f"FATAL ERROR: Failed to load Eye Cascade. Check path: {EYE_CASCADE_PATH}")
    exit()

# --- Load Model ---
try:
    # Loads the Grayscale Model 3 (input shape 48x48x1)
    emotion_classifier = load_model(video_emotion_model_path) 
except Exception as e:
    print(f"FATAL ERROR: Failed to load Keras model from {video_emotion_model_path}: {e}")
    exit()

cv2.namedWindow('Student Attention Detector')

# Initialize counters and map for emotion tally
emotions_map = {e : 0 for e in EMOTIONS} 

# --- Initialize Video Capture ---
if use_live_video:
    cap = cv2.VideoCapture(0)
else:        
    cap = cv2.VideoCapture(video_file_name)
    
if not cap.isOpened():
    print(f"FATAL ERROR: Could not open video source: {'Camera (0)' if use_live_video else video_file_name}")
    cv2.destroyAllWindows()
    exit() 

# Get video properties for time tracking
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0


# --- Main Video Loop ---
while True:
    
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        # Break when end of video is reached or camera stream fails
        print("Video stream finished or camera read failed.")
        break
        
    # Calculate current time for display
    frame_count += 1
    current_second = int(frame_count / frame_rate) if frame_rate > 0 else 0
        
    # Pre-processing
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale used for detection and model input

    # --- Face Detection ---
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    canvas = np.zeros((350, 400, 3), dtype="uint8")
    attentive = False
    
    if len(faces) == 0:
        # Case 1: No face found
        cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Define the face ROIs
        # Use GRAY ROI for both model prediction and eye detection (as model_3 is grayscale)
        roi_gray_input = gray[y:y+h, x:x+w]       
        
        # --- Eye Detection (Attention) ---
        eyes = eye_cascade.detectMultiScale(roi_gray_input)
        for (ex,ey,ew,eh) in eyes[:2]:
            # Draw on the original color frame ROI
            cv2.rectangle(frame[y:y+h, x:x+w], (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            
        if len(eyes) >= 1:
            attentive = True
            
        # --- Emotion Prediction Input Preparation (1 Channel) ---
        if roi_gray_input.size == 0: # Skip if ROI is somehow empty
            continue

        roi_input = cv2.resize(roi_gray_input, (48, 48))
        
        # Apply normalization matching the ImageDataGenerator (rescale=1./255)
        roi_input = roi_input.astype("float") / 255.0
        
        roi_input = img_to_array(roi_input) 
        # roi_input shape is (48, 48, 1) - Correct for grayscale model

        roi_input = np.expand_dims(roi_input, axis=0)
        # roi_input shape is now (1, 48, 48, 1) - MATCHES THE MODEL!

        # --- Prediction ---
        preds = emotion_classifier.predict(roi_input, verbose=0)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        emotions_map[label] += 1
        
        # --- Draw Probability Bars ---
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100) 
            w_bar = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w_bar, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # --- Draw Attention Status on Frame ---
        label_text = f"{'Attentive' if attentive else 'Not-Attentive'} ({label})"
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        # Display current time in the top left of the main frame
        cv2.putText(frame, f"Time: {current_second}s", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


    # --- Display Frames and Check for Exit ---
    cv2.imshow('Student Attention Detector',frame)
    cv2.imshow('Face Emotion Probabilities using AI', canvas) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nExiting application and closing all windows.")