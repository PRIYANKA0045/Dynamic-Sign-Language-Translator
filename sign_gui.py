import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
import os
import pyttsx3
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
from collections import deque

warnings.filterwarnings("ignore")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator (Motion Trail Edition)")
        self.root.geometry("1000x650")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ML and App Variables
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.ml_model = RandomForestClassifier(n_estimators=100) 
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        self.X_train = []
        self.y_train = []
        self.is_trained = False
        
        self.current_sentence = ""
        self.last_prediction = ""
        self.frame_counter = 0
        self.cooldown = 0
        
        # Sequence Tracking Variables
        self.SEQUENCE_LENGTH = 15 
        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.sequence_features = None
        
        # NEW: Visual Trail Buffer (Stores the last 25 positions of the index finger)
        self.trail_buffer = deque(maxlen=25)

        self.cap = cv2.VideoCapture(0)

        self.build_ui()
        self.root.bind('<Key>', self.handle_keypress)
        self.update_video()

    def build_ui(self):
        # Left Frame (Video & Data Collection)
        self.video_frame = ctk.CTkFrame(self.root)
        self.video_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.video_frame, text="Status: Data Collection Mode", text_color="orange", font=("Arial", 18, "bold"))
        self.status_label.pack(pady=5)

        self.prediction_label = ctk.CTkLabel(self.video_frame, text="Predicting: -", font=("Arial", 24))
        self.prediction_label.pack(pady=5)
        
        self.label_entry = ctk.CTkEntry(self.video_frame, placeholder_text="Type word here (e.g., HELLO)", width=250, font=("Arial", 16))
        self.label_entry.pack(pady=10)
        
        ctk.CTkLabel(self.video_frame, text="Press the '+' key to capture a sample for this word", text_color="gray", font=("Arial", 12)).pack()

        # Right Frame (Controls & Text)
        self.control_frame = ctk.CTkFrame(self.root, width=350)
        self.control_frame.pack(side="right", padx=20, pady=20, fill="y")

        ctk.CTkLabel(self.control_frame, text="Translation:", font=("Arial", 20, "bold")).pack(pady=(20, 5))
        self.sentence_display = ctk.CTkTextbox(self.control_frame, height=100, font=("Arial", 24))
        self.sentence_display.pack(padx=20, pady=5, fill="x")
        self.sentence_display.configure(state="disabled")

        ctk.CTkLabel(self.control_frame, text="Controls", font=("Arial", 20, "bold")).pack(pady=(30, 10))
        
        self.btn_speak = ctk.CTkButton(self.control_frame, text="Speak Sentence (ENTER)", command=self.speak_text, fg_color="green", hover_color="darkgreen", state="disabled")
        self.btn_speak.pack(padx=20, pady=10, fill="x")

        self.btn_clear = ctk.CTkButton(self.control_frame, text="Clear Text", command=self.clear_text, state="disabled")
        self.btn_clear.pack(padx=20, pady=5, fill="x")

        ctk.CTkLabel(self.control_frame, text="Model Management", font=("Arial", 16, "bold")).pack(pady=(30, 10))
        
        self.btn_load = ctk.CTkButton(self.control_frame, text="Load Saved Model", command=self.load_model)
        self.btn_load.pack(padx=20, pady=5, fill="x")
        
        self.btn_train = ctk.CTkButton(self.control_frame, text="Train Model", command=self.train_model)
        self.btn_train.pack(padx=20, pady=5, fill="x")

        self.btn_save = ctk.CTkButton(self.control_frame, text="Save Model", command=self.save_model, state="disabled")
        self.btn_save.pack(padx=20, pady=5, fill="x")

        self.btn_reset = ctk.CTkButton(self.control_frame, text="Reset & Retrain", command=self.reset_model, fg_color="red", hover_color="darkred")
        self.btn_reset.pack(padx=20, pady=5, fill="x")

    def extract_features(self, hand_landmarks):
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

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw standard hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # --- NEW: GLOWING MOTION TRAIL LOGIC ---
                    h, w, _ = frame.shape
                    # Get the exact X, Y pixel coordinates of the Index Finger Tip (Landmark 8)
                    index_finger = hand_landmarks.landmark[8] 
                    cx, cy = int(index_finger.x * w), int(index_finger.y * h)
                    self.trail_buffer.append((cx, cy))
                    
                    # Draw the trail from oldest to newest point
                    trail_pts = list(self.trail_buffer)
                    for i in range(1, len(trail_pts)):
                        # Taper the thickness of the line (newer points are thicker)
                        thickness = int(np.sqrt(64 / float(len(trail_pts) - i + 1)) * 2.5)
                        
                        # Draw Cyan "Glow" (Thicker, colored line)
                        cv2.line(frame, trail_pts[i-1], trail_pts[i], (255, 255, 0), thickness + 4)
                        # Draw White "Core" (Thinner, bright center line)
                        cv2.line(frame, trail_pts[i-1], trail_pts[i], (255, 255, 255), thickness)
                    # ---------------------------------------

                    current_features = self.extract_features(hand_landmarks)
                    self.frame_buffer.append(current_features)
                    
                    if len(self.frame_buffer) == self.SEQUENCE_LENGTH:
                        self.sequence_features = np.array(self.frame_buffer).flatten()
                        
                        if self.is_trained:
                            prediction = self.ml_model.predict([self.sequence_features])[0]
                            self.prediction_label.configure(text=f"Predicting: {prediction}")
                            
                            if self.cooldown > 0:
                                self.cooldown -= 1
                            else:
                                if prediction == self.last_prediction:
                                    self.frame_counter += 1
                                    if self.frame_counter == 10: 
                                        if prediction != 'REST': 
                                            self.current_sentence += prediction + " "
                                            self.update_display_text()
                                        self.frame_counter = 0
                                        self.cooldown = 40 
                                else:
                                    self.last_prediction = prediction
                                    self.frame_counter = 0
            else:
                # If the hand leaves the frame, reset everything so it doesn't draw wild lines across the screen
                self.frame_counter = 0
                self.last_prediction = ""
                self.frame_buffer.clear() 
                self.trail_buffer.clear() # Clear the motion trail
                if self.is_trained:
                    self.prediction_label.configure(text="Predicting: -")

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def update_display_text(self):
        self.sentence_display.configure(state="normal")
        self.sentence_display.delete("1.0", "end")
        self.sentence_display.insert("1.0", self.current_sentence)
        self.sentence_display.configure(state="disabled")

    def handle_keypress(self, event):
        key = event.keysym
        
        if self.is_trained:
            if key == "BackSpace":
                words = self.current_sentence.strip().split(" ")
                self.current_sentence = " ".join(words[:-1])
                if len(self.current_sentence) > 0:
                    self.current_sentence += " "
                self.update_display_text()
            elif key == "Return":
                self.speak_text()
                
        elif not self.is_trained and self.sequence_features is not None:
            if key == "plus" or key == "equal": 
                word_label = self.label_entry.get().strip().upper()
                
                if len(word_label) > 0:
                    self.X_train.append(self.sequence_features)
                    self.y_train.append(word_label)
                    self.status_label.configure(text=f"Collected '{word_label}': {self.y_train.count(word_label)} samples")
                else:
                    self.status_label.configure(text="Error: Type a word in the box first!", text_color="red")

    def speak_text(self):
        if len(self.current_sentence.strip()) > 0:
            text_to_speak = self.current_sentence
            threading.Thread(target=lambda: self._speak_thread(text_to_speak)).start()
            self.clear_text()

    def _speak_thread(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def clear_text(self):
        self.current_sentence = ""
        self.update_display_text()

    def load_model(self):
        if os.path.exists("sign_data_words.pkl"):
            with open("sign_data_words.pkl", "rb") as f:
                self.X_train, self.y_train = pickle.load(f)
            self.ml_model.fit(self.X_train, self.y_train)
            self.activate_model()
            self.status_label.configure(text="Model Loaded Successfully!", text_color="lightgreen")
        else:
            self.status_label.configure(text="Error: sign_data_words.pkl not found", text_color="red")

    def train_model(self):
        if len(self.X_train) > 0:
            self.ml_model.fit(self.X_train, self.y_train)
            self.activate_model()
            self.status_label.configure(text="Model Trained! Start Signing.", text_color="lightgreen")
            self.label_entry.configure(state="disabled") 
        else:
            self.status_label.configure(text="No data collected yet!", text_color="red")

    def activate_model(self):
        self.is_trained = True
        self.btn_load.configure(state="disabled")
        self.btn_train.configure(state="disabled")
        self.btn_save.configure(state="normal")
        self.btn_speak.configure(state="normal")
        self.btn_clear.configure(state="normal")

    def save_model(self):
        with open("sign_data_words.pkl", "wb") as f:
            pickle.dump((self.X_train, self.y_train), f)
        self.status_label.configure(text="Model Saved!", text_color="lightgreen")

    def reset_model(self):
        self.is_trained = False
        self.X_train.clear()
        self.y_train.clear()
        self.clear_text()
        self.trail_buffer.clear()
        
        self.btn_load.configure(state="normal")
        self.btn_train.configure(state="normal")
        self.btn_save.configure(state="disabled")
        self.btn_speak.configure(state="disabled")
        self.btn_clear.configure(state="disabled")
        self.label_entry.configure(state="normal")
        self.status_label.configure(text="Status: Data Collection Mode", text_color="orange")

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()