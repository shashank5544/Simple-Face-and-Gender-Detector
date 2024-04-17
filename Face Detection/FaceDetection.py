import cv2
import numpy as np

# Initialize frame size
frame_width = 1280
frame_height = 720

# Load the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(
    r'Path To gender_deploy.prototxt', 
    r'Path To gender_net.caffemodel')

# Define the list of gender types the model recognizes
gender_list = ['Male', 'Female']

# Define the mean values for the ImageNet training set
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_face_and_gender(img):
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the face ROI
        face_img = img[y:y+h, h:h+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)

         # Put gender text above the rectangle
        font_scale = 2.5 if gender == 'Male' else 2.0
        font_color = (0, 0, 255) if gender == 'Male' else (0, 255, 0)
        cv2.putText(img, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2)

         # Overlay the text multiple times to create a bold effect
        thickness = 3
        for i in range(thickness):
            cv2.putText(img, gender, (x+i, y-10+i), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2)


    # Resize the image
    resized_img = cv2.resize(img, (800, 600))
    # Display the output
    cv2.imshow('img', resized_img)
    cv2.waitKey()

# Test the function
img = cv2.imread('man.jpg')
detect_face_and_gender(img)
