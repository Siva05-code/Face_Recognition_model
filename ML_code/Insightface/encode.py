import os
import cv2
import pickle
from tqdm import tqdm
import numpy as np
from insightface.app import FaceAnalysis

def encode_faces_insightface(dataset_path, output_path="model/2023_27_AIML_dummy.pickle"):
    # Initialize FaceAnalysis with GPU and ArcFace model
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    known_encodings = []
    known_names = []

    # Iterate over each student folder
    for student in tqdm(os.listdir(dataset_path), desc="Processing Students"):
        student_folder = os.path.join(dataset_path, student)
        if not os.path.isdir(student_folder):
            continue

        for image_name in os.listdir(student_folder):
            image_path = os.path.join(student_folder, image_name)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load {image_path}")
                    continue

                faces = app.get(image)
                if len(faces) == 0:
                    # print(f"No face found in {image_path}")
                    continue

                # Use the first detected face
                face = faces[0]
                embedding = face.embedding

                known_encodings.append(embedding)
                known_names.append(student)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Save encodings
    data = {"encodings": known_encodings, "names": known_names}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print("ArcFace encoding completed and saved to:", output_path)
encode_faces_insightface('/Users/sivakarthick/Downloads/ML2_miniprj/Dataset/2023_27_AIML_augmented','model/2023_27_AIML_dummy.pickle')
