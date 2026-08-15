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
        
        # Visual Trail Buffer
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
        
        # --- NEW: Added a clickable button so you don't have to rely on the keyboard! ---
        self.btn_capture = ctk.CTkButton(self.video_frame, text="Capture Sample (+)", command=self.capture_sample)
        self.btn_capture.pack(pady=5)

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
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # GLOWING MOTION TRAIL LOGIC
                    h, w, _ = frame.shape
                    index_finger = hand_landmarks.landmark[8] 
                    cx, cy = int(index_finger.x * w), int(index_finger.y * h)
                    self.trail_buffer.append((cx, cy))
                    
                    trail_pts = list(self.trail_buffer)
                    for i in range(1, len(trail_pts)):
                        thickness = int(np.sqrt(64 / float(len(trail_pts) - i + 1)) * 2.5)
                        cv2.line(frame, trail_pts[i-1], trail_pts[i], (255, 255, 0), thickness + 4)
                        cv2.line(frame, trail_pts[i-1], trail_pts[i], (255, 255, 255), thickness)

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
                self.frame_counter = 0
                self.last_prediction = ""
                self.frame_buffer.clear() 
                self.trail_buffer.clear() 
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

    # --- NEW: Dedicated Capture Function that auto-cleans typos ---
    def capture_sample(self):
        if not self.is_trained and self.sequence_features is not None:
            # 1. Grab text and remove any accidental + or = signs
            raw_text = self.label_entry.get().upper()
            clean_word = raw_text.replace("=", "").replace("+", "").strip()
            
            # 2. Update the text box so you don't see the = signs piling up!
            self.label_entry.delete(0, 'end')
            self.label_entry.insert(0, clean_word)
            
            # 3. Save the clean word to the AI
            if len(clean_word) > 0:
                self.X_train.append(self.sequence_features)
                self.y_train.append(clean_word)
                self.status_label.configure(text=f"Collected '{clean_word}': {self.y_train.count(clean_word)} samples")
            else:
                self.status_label.configure(text="Error: Type a word in the box first!", text_color="red")

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
                # We use a slight delay so the GUI can finish typing the '=' before we wipe it out
                self.root.after(10, self.capture_sample)

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
                saved_X, saved_y = pickle.load(f)
            
            # Merge old data with any new data you just collected, preventing duplicates
            if not getattr(self, 'data_loaded', False):
                self.X_train = saved_X + self.X_train
                self.y_train = saved_y + self.y_train
                self.data_loaded = True
                
            self.ml_model.fit(self.X_train, self.y_train)
            self.activate_model()
            
            # Show you exactly which words are currently loaded!
            unique_words = list(set(self.y_train))
            self.status_label.configure(text=f"Loaded {len(unique_words)} words: {', '.join(unique_words)}", text_color="lightgreen")
        else:
            self.status_label.configure(text="Error: No saved data found", text_color="red")

    def save_model(self):
        # Prevent wiping out old data if you forgot to click "Load" first
        if os.path.exists("sign_data_words.pkl") and not getattr(self, 'data_loaded', False):
            with open("sign_data_words.pkl", "rb") as f:
                saved_X, saved_y = pickle.load(f)
            self.X_train = saved_X + self.X_train
            self.y_train = saved_y + self.y_train
            self.data_loaded = True

        with open("sign_data_words.pkl", "wb") as f:
            pickle.dump((self.X_train, self.y_train), f)
            
        unique_words = list(set(self.y_train))
        self.status_label.configure(text=f"Saved! AI now knows {len(unique_words)} words.", text_color="lightgreen")

    def reset_model(self):
        self.is_trained = False
        self.data_loaded = False  # Reset the merge lock
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
        self.btn_capture.configure(state="normal") 
        self.status_label.configure(text="Status: Data Collection Mode", text_color="orange")

    def train_model(self):
        if len(self.X_train) > 0:
            self.ml_model.fit(self.X_train, self.y_train)
            self.activate_model()
            self.status_label.configure(text="Model Trained! Start Signing.", text_color="lightgreen")
            self.label_entry.configure(state="disabled") 
            self.btn_capture.configure(state="disabled") # Disable capture button
        else:
            self.status_label.configure(text="No data collected yet!", text_color="red")

    def activate_model(self):
        self.is_trained = True
        self.btn_load.configure(state="disabled")
        self.btn_train.configure(state="disabled")
        self.btn_save.configure(state="normal")
        self.btn_speak.configure(state="normal")
        self.btn_clear.configure(state="normal")

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()