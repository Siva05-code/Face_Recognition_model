import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# Predefined color list (distinct colors)
COLORS = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 255),   # Magenta
]

# 1. Recognize Faces and Draw Labels
def recognize_faces(image,encodings_path='/Users/sivakarthick/Downloads/ML2_miniprj/ML_code/Insightface/model/2023_27_AIML1.pickle',threshold=0.45):
    # Load stored encodings
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

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
        # for idx , line in enumerate(name[::-1]):
        #     cv2.putText(image, line, (x1, y1-20-idx*1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        recognized.append(name)
    return image, recognized


# 2. Single Image
def run_with_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not load image:", image_path)
        return
    frame, names = recognize_faces(frame)

    # Display sorted unique roll numbers 
    recognized_sorted = sorted(set(names))
    print("\n✅ Recognized Roll Numbers:")
    for name in recognized_sorted:
        print(name)
    print("\nTotal Recognized:", len(recognized_sorted))

    # Display the frame
    cv2.imshow("Image Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 3. Folder
def run_on_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {filename}")
            run_with_image(image_path)

# 4. Webcam
def run_with_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    n=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        result_frame, names = recognize_faces(frame)
        n.append(names)
        # Show webcam feed
        cv2.imshow("Webcam Recognition (Press q to quit)", result_frame)

        # Optional: Print roll numbers continuously (comment if not needed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    recognized_sorted = sorted(set(i for i in n))
    print("Live Recognized:", recognized_sorted)
    cap.release()
    cv2.destroyAllWindows()

# 5. Video
from collections import Counter
import cv2

def process_video(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    name_counts = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame directly
        if frame_count % frame_interval == 0:
            # Detect & recognize faces directly on frame
            result_frame, names = recognize_faces(frame)

            # Count occurrences for stability
            for n in names:
                name_counts[n] += 1

            # Show video with bounding boxes
            cv2.imshow("Video Attendance", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Keep names detected at least 3 times
    final_names = sorted([n for n, c in name_counts.items() if c >= 1])

    print("\n✅ Recognized Roll Numbers from Video:")
    for name in final_names:
        print(name)
    print("\nTotal Recognized:", len(final_names))

    return final_names



# 6. Interface
def main():
    print("\nFace Recognition Interface (InsightFace)")
    print("1 - Webcam")
    print("2 - Single Image")
    print("3 - Folder of Images")
    print("4 - Video")
    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        run_with_webcam()
    elif choice == '2':
        path = input("Enter image path: ")
        run_with_image(path)
    elif choice == '3':
        folder_path = input("Enter folder path: ")
        run_on_folder(folder_path)
    elif choice == "4":
        video_path = input("Enter video path: ")
        process_video(video_path)
    else:
        print("Invalid choice.")

main()
