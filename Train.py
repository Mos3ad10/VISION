import os
import cv2
import numpy as np

# -------------------------------
# Initialize Face Detector
# -------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -------------------------------
# Detect Face Function
# -------------------------------
def detect_face(gray_img):
    faces = face_detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20)
    )
    return faces

# -------------------------------
# Training Execution
# -------------------------------
def train_model():
    dataset_path = "./dataset/waled"
    faces = []
    labels = []

    label = 0  # 0 corresponds to the person in the "waled" folder

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        return

    for image_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, image_name)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = detect_face(gray)

        if len(detected_faces) == 0:
            # FIX: Skip the image instead of using the whole background
            # Using the whole image ruins the accuracy of the recognizer.
            print(f"⚠️ No face detected, skipping: {img_path}")
            continue

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(label)
            print(f"✅ Face added: {img_path}")

    if len(faces) == 0:
        print("❌ Error: No faces were collected. Cannot train model.")
        return

    faces = np.array(faces)
    labels = np.array(labels)

    print(f"\nTotal faces collected: {len(faces)}")

    # -------------------------------
    # Train LBPH Model
    # -------------------------------
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    model.train(faces, labels)
    model.save("face_model.yml")
    print("🎯 Training completed and model saved as face_model.yml")

# This ensures training only runs if you execute Train.py directly
if __name__ == "__main__":
    train_model()