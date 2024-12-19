import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from deepface import DeepFace

# Paths
db_path = "dataset"  # Directory containing known faces
test_path = "test_data"  # Directory containing test images

# Initialize variables
true_labels = []
predicted_labels = []

def predict_face(img_path):
    """
    Function to predict face using DeepFace.
    Returns the recognized name or 'Unknown'.
    """
    try:
        result = DeepFace.find(img_path=img_path, db_path=db_path, model_name="Facenet", enforce_detection=False)
        if len(result) > 0:
            # Extract the recognized name from the identity path
            recognized_name = result[0]["identity"][0].split(os.sep)[-2]
            return recognized_name
        else:
            return "Unknown"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

# Load Test Data
print("Starting Evaluation...")
for person_name in os.listdir(test_path):  # Iterate through each person's folder
    person_folder = os.path.join(test_path, person_name)

    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):  # Iterate through each image
            img_path = os.path.join(person_folder, img_name)

            # Predict the face
            predicted_name = predict_face(img_path)

            # Store the true label and predicted label
            true_labels.append(person_name)
            predicted_labels.append(predicted_name)

            print(f"Image: {img_name}, True: {person_name}, Predicted: {predicted_name}")

# Generate Confusion Matrix and Performance Metrics
print("\nEvaluation Results:")
print("Confusion Matrix:")
labels = list(set(true_labels)) + ["Unknown"]
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
print(cm)

# Print Performance Metrics
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=labels))

# Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"\nOverall Accuracy: {accuracy:.2f}")
