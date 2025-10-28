from decode import recognize_faces, calculate_recognition_accuracy
import cv2
import os
import onnxruntime

# 1. Single Image
def run_with_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not load image:", image_path)
        return
    frame, names = recognize_faces(frame)

    # Display sorted unique roll numbers 
    recognized_sorted = sorted(set(names))
    print("\n   Recognized Roll Numbers:")
    for name in recognized_sorted:
        print(name)
    print("\nTotal Recognized:", len(recognized_sorted))
    # Calculate recognition accuracy on a test dataset
    dataset_path = "/Users/sivakarthick/Downloads/ML2_miniprj/Dataset/2023_27_AIML_augmented"
    encodings_path = "/Users/sivakarthick/Downloads/ML2_miniprj/ML_code/Insightface/model/2023_27_AIML1.pickle"
    calculate_recognition_accuracy(dataset_path, encodings_path)

    # Display the frame
    cv2.imshow("Image Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. Folder
def run_on_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {filename}")
            run_with_image(image_path)

# 3. Webcam
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

#  Interface
def main():
    print("\nFace Recognition Interface (InsightFace)")
    print("1 - Webcam")
    print("2 - Single Image")
    print("3 - Folder of Images")
    choice = input("Enter choice (1/2/3): ")

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
