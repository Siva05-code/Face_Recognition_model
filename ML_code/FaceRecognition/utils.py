import face_recognition
import cv2
import pickle

def recognize_faces_from_frame(frame, encoding_path="/Users/sivakarthick/Downloads/ML2_miniprj/ML_code/FaceRecognition/models/encodings1.pickle"):
    with open(encoding_path, "rb") as f:
        data = pickle.load(f)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.46)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)
    sorted_rolls = sorted(set(names))
    print("Present Students (Sorted):")
    for roll in sorted_rolls:
        print(roll)
    return boxes, names
