# 🤟 Dynamic Sign Language Translator

A real-time, spatiotemporal Sign Language Translator built with Python, OpenCV, and MediaPipe. Unlike static gesture recognizers, this application uses a sliding-window memory buffer to understand **moving signs** and full words, complete with a modern GUI, glowing motion trails, and Text-to-Speech (TTS).

## ✨ Features

* **Spatiotemporal AI:** Uses a 15-frame rolling memory buffer and a `RandomForestClassifier` to understand complex, moving hand gestures—not just static poses.
* **Teachable Interface:** Train custom words on the fly. You don't need to download massive datasets; teach the AI your own signs instantly through the GUI.
* **Motion Trails:** Visual feedback using OpenCV to draw a glowing, tapering line following the index finger, allowing users to "paint" and see the shapes of the signs they are training.
* **Text-to-Speech (TTS):** Integrated `pyttsx3` engine speaks the translated sentences out loud.
* **Modern GUI:** Built with `CustomTkinter` for a sleek, dark-mode desktop application experience.

## 🛠️ Tech Stack

* **Computer Vision:** `opencv-python`, `mediapipe`
* **Machine Learning:** `scikit-learn` (Random Forest Classifier), `numpy`
* **Interface:** `customtkinter`, `Pillow`
* **Audio:** `pyttsx3`

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/PRIYANKA0045/dynamic-sign-translator.git](https://github.com/PRIYANKA0046/dynamic-sign-translator.git)
   cd dynamic-sign-translator
