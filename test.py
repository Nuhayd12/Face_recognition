import cv2
import os
from deepface import DeepFace

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Database directory
db_path = "dataset"
print("Initializing Face Recognition...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        temp_path = "temp_face.jpg"

        # Save detected face temporarily
        cv2.imwrite(temp_path, face_img)

        try:
            # Recognize face using DeepFace
            result = DeepFace.find(img_path=temp_path, db_path=db_path, model_name="Facenet", enforce_detection=False)
            
            if len(result) > 0:
                recognized_name = result[0]["identity"][0].split(os.sep)[-2]
                confidence = round(1 - result[0]["distance"][0], 2)
            else:
                recognized_name = "Unknown"
                confidence = 0

        except Exception as e:
            print(f"Error: {e}")
            recognized_name = "Error"
            confidence = 0

        # Draw results on the frame
        display_text = f"{recognized_name} ({confidence*100:.1f}%)" if confidence > 0 else recognized_name
        cv2.putText(frame, display_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Face recognition stopped.")
video.release()
cv2.destroyAllWindows()
