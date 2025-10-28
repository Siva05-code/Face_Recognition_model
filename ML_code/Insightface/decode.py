import cv2
import pickle
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime

# Predefined color list (distinct colors)
COLORS = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 255),   # Magenta
]

# Recognize Faces and Draw Labels
def recognize_faces(image,encodings_path='/Users/sivakarthick/Downloads/ML2_miniprj/ML_code/Insightface/model/2023_27_AIML1.pickle',threshold=0.5):
    # Load stored encodings
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640)) # 0 for GPU, -1 for CPU (depending on backend)


    faces = app.get(image)
    recognized = []

    if not faces:
        print("No faces detected.")
        return image, recognized
    c=1
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        embedding = face.embedding.reshape(1, -1)

        if known_encodings:
            sims = cosine_similarity(embedding, known_encodings)[0]
            max_index = np.argmax(sims)
            max_score = sims[max_index]

            if max_score >= threshold:
                name = known_names[max_index]
            else:
                name = "unknown"+str(c)
                c+=1
        else:
            name = "unknown"+str(c)
            c+=1

        # Assign different color per face
        color = COLORS[i % len(COLORS)]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw name vertically above face
        # text_lines = name.split(" ")
        # for idx, line in enumerate(name[::-1]):
        cv2.putText( image,name,(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,0.7,color, 2)
        recognized.append(name)
    return image, recognized

def calculate_recognition_accuracy(dataset_path, encodings_path, threshold=0.5):
    """
    Evaluate recognition accuracy over a dataset organized as:
    dataset_path/
       ├── person1/
       │     ├── img1.jpg
       │     ├── img2.jpg
       ├── person2/
             ├── img1.jpg
             ├── img2.jpg
    """
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

    total = 0
    correct = 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            faces = app.get(image)
            if not faces:
                continue

            embedding = faces[0].embedding.reshape(1, -1)
            sims = cosine_similarity(embedding, known_encodings)[0]
            max_index = np.argmax(sims)
            max_score = sims[max_index]

            predicted_name = known_names[max_index] if max_score >= threshold else "unknown"

            if predicted_name == person_name:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n✅ Recognition Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")
    return accuracy


def calculate_recognition_accuracy1(dataset_path, encodings_path, threshold=0.5):
    """
    Evaluate recognition accuracy over a dataset organized as:
    dataset_path/
       ├── person1.png
       ├── person2.png
       ├── person3.jpg
    where each image filename corresponds to the person's name.
    Example:  'sivakarthick.png' → true label = 'sivakarthick'
    """
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

    total = 0
    correct = 0

    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        if not os.path.isfile(img_path):
            continue

        # Extract ground truth label (filename without extension)
        person_name = os.path.splitext(img_name)[0]

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        faces = app.get(image)
        if not faces:
            print(f"⚠️ No face detected in {img_path}")
            continue

        embedding = faces[0].embedding.reshape(1, -1)
        sims = cosine_similarity(embedding, known_encodings)[0]
        max_index = np.argmax(sims)
        max_score = sims[max_index]

        predicted_name = known_names[max_index] if max_score >= threshold else "unknown"

        if predicted_name == person_name:
            correct += 1
        else:
            print(f"❌ Wrong match: {img_name} → predicted as {predicted_name}")
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n✅ Recognition Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")
    return accuracy
