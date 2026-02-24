import cv2
import os
from Train import detect_face


def test_image():
    # Load the trained model
    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read("face_model.yml")
    except cv2.error:
        print("❌ Error: Could not load 'face_model.yml'. Did you run Train.py first?")
        return

    # Label mapping
    label_dict = {
        0: "Waled",  # Updated to match the folder name you trained on
        1: "Person2"
    }

    # Full path to the test image
    image_path = "./test_images/WhatsApp Image 2026-02-23 at 11.21.50 PM.jpeg"

    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # Read image
    test_img = cv2.imread(image_path)
    if test_img is None:
        print("❌ Cannot read test image.")
        return

    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detect_face(gray)

    if len(faces) == 0:
        print("⚠️ No faces detected in the test image.")

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        # Note: In LBPH, lower confidence means a closer match (distance).
        if confidence < 70:
            name = label_dict.get(label, "Unknown")
        else:
            name = "Unknown"

        cv2.putText(test_img, f"{name} ({confidence:.1f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show result
    output_path = "output_result.jpeg"
    cv2.imwrite(output_path, test_img)
    print(f"✅ Success! Result saved to {output_path}")


if __name__ == "__main__":
    test_image()