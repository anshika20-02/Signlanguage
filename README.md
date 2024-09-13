Hand Gesture Recognition System

Overview:
This project is a hand gesture recognition system that uses computer vision and machine learning to recognize and classify hand gestures. The system consists of four main components:

1. Data Collection: A script to collect images of hand gestures using a webcam.
2. Data Preprocessing: A script to preprocess the collected images and extract features using MediaPipe.
3. Model Training: A script to train a machine learning model using the preprocessed data.
4. Real-time Recognition: A script to recognize hand gestures in real-time using the trained model.


Getting Started

Prerequisites:
Python 3.x
OpenCV
MediaPipe
scikit-learn


Installation:
Install the required packages: pip install -r requirements.txt
Running the System
Step 1: Collect data: python collect_imgs.py
Step 2: Preprocess data: python create_dataset.py
Step 3: Train the model: python train_classifier.py
Step 4: Run the real-time recognition: python final.py

Scripts
collect_imgs.py
This script collects images of hand gestures using a webcam. It creates a directory for each class (e.g. A, B, C, etc.) and saves the images in the corresponding directory.

create_dataset.py
This script preprocesses the collected images and extracts features using MediaPipe. It creates a pickle file containing the preprocessed data and labels.

train_classifier.py
This script trains a machine learning model using the preprocessed data. It saves the trained model to a pickle file.

final.py
This script recognizes hand gestures in real-time using the trained model. It displays the recognized gesture on the screen.


Acknowledgments
This project uses the following libraries and frameworks:

OpenCV
MediaPipe
scikit-learn
Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

Issues
If you encounter any issues or have questions, please open an issue on the GitHub repository.