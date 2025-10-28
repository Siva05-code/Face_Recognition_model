import os
import pickle
import face_recognition
def encode_faces(dataset_path):
    known_encodings = []
    known_names = []

    for student in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student)
        if not os.path.isdir(student_folder):
            continue

        for image_name in os.listdir(student_folder):
            image_path = os.path.join(student_folder, image_name)
            image = face_recognition.load_image_file(image_path)

            boxes = face_recognition.face_locations(image, model="cnn")
            encodings = face_recognition.face_encodings(image, boxes)

            if len(encodings) == 0:
                print(f"Skipping {image_path}, no face found.")
                continue

            known_encodings.append(encodings[0])
            known_names.append(student)

    data = {"encodings": known_encodings, "names": known_names}

    # Ensure the folder exists
    os.makedirs("models", exist_ok=True)

    # Path to the file
    file_path = "models/encodings1.pickle"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    else:
        with open("models/encodings1.pickle", "wb") as f:
            pickle.dump(data, f)

    print("Encoding completed.")
encode_faces('/Users/sivakarthick/Downloads/ML2_miniprj/Dataset/custom_dataset_cropped1')
