import cv2
import os
from deepface import DeepFace

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create the dataset directory if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Input for person name
name = input("Enter Your Name: ")
person_dir = f"dataset/{name}"
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

i = 0  # Counter for saved images

def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Apply histogram equalization
    return equalized

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess the face (grayscale and histogram equalization)
        processed_face = preprocess_face(face_img)
        resized_face = cv2.resize(processed_face, (50, 50))

        # Save face images every 10 frames
        if i % 10 == 0 and i // 10 < 100:
            face_path = os.path.join(person_dir, f"face_{i//10}.jpg")
            cv2.imwrite(face_path, resized_face)
            print(f"Saved {face_path}")
        
        i += 1

        # Display the progress
        cv2.putText(frame, f"Collecting: {i//10}/100", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or i // 10 >= 100:
        break

print("Data collection completed!")
video.release()
cv2.destroyAllWindows()
