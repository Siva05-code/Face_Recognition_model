import cv2
from utils import recognize_faces_from_frame

def draw_faces(frame, boxes, names):
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def run_with_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        boxes, names = recognize_faces_from_frame(frame)

        frame = draw_faces(frame, boxes, names)
        cv2.imshow("Webcam Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import os
import cv2

def run_with_image(image_path):
    frame = cv2.imread(image_path)
    boxes, names = recognize_faces_from_frame(frame)
    frame = draw_faces(frame, boxes, names)

    cv2.imshow("Image Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {image_path}")
            run_with_image(image_path)

def main():
    choice = input("Enter 1 for Webcam or 2 to Upload Image or 3: ")

    if choice == '1':
        run_with_webcam()
    elif choice == '2':
        path = input("Enter image path: ")
        run_with_image(path)
    elif choice == '3':
        folder_path = input("Enter folder path: ")
        run_on_folder(folder_path)
    else:
        print("Invalid choice.")

main()