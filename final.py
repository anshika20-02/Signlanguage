# Importing Libraries
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from fpdf import FPDF
import pyttsx3  # Import pyttsx3 for text-to-speech
import winsound  # For beep sound (Windows-specific)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# List all available voices and select an Indian English voice
voices = engine.getProperty('voices')
for voice in voices:
    if 'India' in voice.name or 'Indian' in voice.name or 'English (India)' in voice.name:
        engine.setProperty('voice', voice.id)
        print(f"Selected voice: {voice.name}")
        break

# Loading the Pre-trained Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start capturing video from the webcam. 0 refers to the default webcam.
cap = cv2.VideoCapture(0)

# Initializing mediapipe hand modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {i: chr(65 + i) for i in range(26)}

# Initialize global variables
predicted_letters = []
current_word = []
letter_to_add = None
frame_count = 0
frames_for_sign = 15
last_sign_time = time.time()
sign_timeout = 2
confirmed_letter = None
predicted_character = ' '

# Track the session start time
start_time = time.time()

def add_letter_to_word(letter):
    global current_word
    current_word.append(letter)
    # Add a beep sound to give auditory feedback when a letter is confirmed
    winsound.Beep(1000, 200)  # Frequency 1000Hz, duration 200ms

def add_word_to_sentence():
    global predicted_letters, current_word
    if current_word:
        predicted_letters.append(''.join(current_word))
        current_word = []
        # Play a longer beep when a word is added
        winsound.Beep(1500, 300)  # Frequency 1500Hz, duration 300ms
        speak_sentence()  # Speak the sentence whenever a new word is added

def clear_word():
    global current_word
    current_word = []

def clear_sentence():
    global predicted_letters
    predicted_letters = []
    global current_word
    current_word = []

def get_word():
    global current_word
    return ''.join(current_word)

def get_sentence():
    global predicted_letters
    return ' '.join(predicted_letters)

def clear_all():
    clear_sentence()
    clear_word()

def save_to_text_file():
    with open("predicted_text.txt", "w") as file:
        file.write(get_sentence())

def save_to_pdf_file():
    total_time = time.time() - start_time  # Calculate total time
    total_letters = sum(len(word) for word in predicted_letters)
    total_words = len(predicted_letters)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, get_sentence())

    # Add analytics section in the PDF
    pdf.ln(10)  # Line break
    pdf.cell(200, 10, txt="Analytics Summary", ln=True, align='C')
    pdf.ln(5)  # Line break
    pdf.cell(200, 10, txt=f"Total Letters Recognized: {total_letters}", ln=True)
    pdf.cell(200, 10, txt=f"Total Words Formed: {total_words}", ln=True)
    pdf.cell(200, 10, txt=f"Time Spent on Recognition: {total_time:.2f} seconds", ln=True)

    pdf.output("predicted_text_with_analytics.pdf")

def speak_sentence():
    sentence = get_sentence()
    engine.say(sentence)
    engine.runAndWait()

while True:
    data_aux = []

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image from the webcam.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

        x_min = min(x_)
        y_min = min(y_)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - x_min)
            data_aux.append(y - y_min)

        data_aux = data_aux[:42]

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        if letter_to_add == predicted_character:
            frame_count += 1
        else:
            letter_to_add = predicted_character
            frame_count = 1

        if frame_count >= frames_for_sign:
            confirmed_letter = predicted_character
            letter_to_add = None
            frame_count = 0
            last_sign_time = time.time()

            if confirmed_letter == '<CLR>':
                clear_all()
            elif confirmed_letter == ' ':
                add_word_to_sentence()
                clear_word()
            else:
                add_letter_to_word(confirmed_letter)

            confirmed_letter = None

    else:
        if time.time() - last_sign_time > sign_timeout:
            add_word_to_sentence()
            clear_word()

        predicted_character = ' '

    sentence = get_sentence()
    word = get_word()
    cv2.putText(frame, f"Predicted Letter: {predicted_character}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Word: {word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Sentence: {get_sentence()}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        clear_all()
    elif key == ord('t'):  # Press 't' to save to text file
        save_to_text_file()
    elif key == ord('p'):  # Press 'p' to save to PDF file
        save_to_pdf_file()
    elif key == ord('s'):  # Press 's' to speak the current sentence
        speak_sentence()

cap.release()
cv2.destroyAllWindows()
