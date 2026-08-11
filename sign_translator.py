#This file have simple Interface which is provided by OpenCV for interactive UI use sign_gui.py


import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import warnings
import pickle
import os
from spellchecker import SpellChecker
import pyttsx3 # NEW: Text-to-Speech library
import threading # NEW: Allows speech without freezing the camera

warnings.filterwarnings("ignore")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# Initialize Machine Learning, Spellchecker, and TTS Engine
knn_model = KNeighborsClassifier(n_neighbors=3)
spell = SpellChecker()
X_train = [] 
y_train = [] 
is_trained = False

# Initialize the Voice Engine
tts_engine = pyttsx3.init()
# Optional: Slow down the talking speed slightly so it sounds more natural
tts_engine.setProperty('rate', 150) 

def speak_text(text):
    """ Runs the text-to-speech engine in the background """
    def run_speech():
        tts_engine.say(text)
        tts_engine.runAndWait()
    # Using a thread prevents the webcam video from freezing while talking
    threading.Thread(target=run_speech).start()

# Word Building Variables
current_word = ""
last_prediction = ""
frame_counter = 0
FRAMES_TO_CONFIRM = 15 
cooldown = 0 

def extract_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - wrist.x)
        features.append(lm.y - wrist.y)
        features.append(lm.z - wrist.z)
        
    features = np.array(features)
    max_val = np.max(np.abs(features))
    if max_val > 0:
        features = features / max_val
        
    return features

# Start Video Capture
cap = cv2.VideoCapture(0)

print("--- VIRTUAL SIGN LANGUAGE TRANSLATOR ---")
print("[DATA COLLECTION MODE]")
print("A-Z / 0-9 : Collect data (Use '0' for resting hand)")
print("L         : Load saved model")
print("ENTER     : Train the model")
print("ESC       : Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    features = None
    suggestion = "" 
    
    if is_trained:
        words_list = current_word.split(" ")
        active_word = words_list[-1] 
        
        if len(active_word) > 0:
            corr = spell.correction(active_word)
            if corr:
                suggestion = corr.upper()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_features(hand_landmarks)
            
            if is_trained:
                prediction = knn_model.predict([features])[0]
                
                if cooldown > 0:
                    cooldown -= 1
                else:
                    if prediction == last_prediction:
                        frame_counter += 1
                        if frame_counter == FRAMES_TO_CONFIRM:
                            if prediction != '0':
                                current_word += prediction
                            frame_counter = 0
                            cooldown = 20 
                    else:
                        last_prediction = prediction
                        frame_counter = 0
                
                cv2.putText(frame, f"Predicting: {prediction}", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
    else:
        frame_counter = 0
        last_prediction = ""

    # UI for the Typed Sentence & Suggestion
    if is_trained:
        cv2.putText(frame, f"Sentence: {current_word}", (50, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        if suggestion and suggestion != active_word.upper():
            cv2.putText(frame, f"Suggestion (Press TAB): {suggestion}", (50, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

    if not is_trained:
        cv2.putText(frame, f"Data: {len(X_train)} samples", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(X_train) == 0:
            cv2.putText(frame, "Press Keys to train, or 'L' to Load.", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press ENTER when ready to train.", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Model Active! 'S' Save | 'R' Reset", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "Space | Backspace | Tab | ENTER (Speak)", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Sign Language Translator", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: 
        break
        
    elif not is_trained:
        if key == 13: 
            if len(X_train) > 0:
                print("\nTraining model...")
                knn_model.fit(X_train, y_train)
                is_trained = True
                print("Model trained! Start signing.")
            else:
                print("No data to train on!")
                
        elif (key == ord('l') or key == ord('L')) and len(X_train) == 0:
            if os.path.exists("sign_data.pkl"):
                with open("sign_data.pkl", "rb") as f:
                    X_train, y_train = pickle.load(f)
                knn_model.fit(X_train, y_train)
                is_trained = True
                print("\nSUCCESS: Model loaded! Start signing.")
            else:
                print("\nERROR: 'sign_data.pkl' not found.")
                
        elif key != 255:
            if (65 <= key <= 90) or (97 <= key <= 122) or (48 <= key <= 57):
                if features is not None:
                    char = chr(key).upper() 
                    X_train.append(features)
                    y_train.append(char)
                    print(f"Collected data for: {char} (Total: {y_train.count(char)})")

    else: 
        if key == ord('r') or key == ord('R'): 
            is_trained = False
            X_train.clear()
            y_train.clear()
            current_word = ""
            print("\nModel reset. Ready to collect data.")
            
        elif key == ord('s') or key == ord('S'): 
            with open("sign_data.pkl", "wb") as f:
                pickle.dump((X_train, y_train), f)
            print("\nModel saved successfully!")
            
        elif key == ord('c') or key == ord('C'): 
            current_word = ""
            
        elif key == 32: 
            current_word += " "
            
        elif key == 8 or key == 127: 
            current_word = current_word[:-1]
            
        elif key == 9: 
            if suggestion and len(words_list) > 0:
                words_list[-1] = suggestion
                current_word = " ".join(words_list) + " "
                
        # --- NEW: ENTER KEY TO SPEAK ---
        elif key == 13: # ENTER key
            if len(current_word.strip()) > 0:
                print(f"Speaking: {current_word}")
                speak_text(current_word) # Trigger the voice
                current_word = "" # Clear the screen automatically

cap.release()
cv2.destroyAllWindows()