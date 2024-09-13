#collect_imgs.py
#collect all images
import os

import cv2

# Setting Up the Data Directory:
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Specifying the Number of Classes and Dataset Size:
number_of_classes = 26      # 26 different classes (probably for each letter of the alphabet).
dataset_size = 100          #100 images will be collected for each class.


# opens the webcam for capturing video frames.
cap = cv2.VideoCapture(0)

# Creating Folders for Each Class:
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        # Starting Image Collection:
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Releasing the Webcam and Closing Windows
cap.release()
cv2.destroyAllWindows()
