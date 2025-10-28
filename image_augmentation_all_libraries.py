import os
import cv2
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imageio
import imgaug.augmenters as iaa
from torchvision import transforms
import tensorflow as tf

# Albumentations
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(p=0.3),
    A.Resize(256, 256),
])

# imgaug
imgaug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.LinearContrast((0.75, 1.5))
])

# torchvision
torchvision_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5)
])

# Keras
keras_augment = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 256),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.5),
    tf.keras.layers.Rescaling(1.0)
])

# OpenCV augment
def opencv_augment(image):
    image = cv2.resize(image, (256, 256))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols // 2, rows // 2), np.random.randint(-20, 20), 1)
    image = cv2.warpAffine(image, matrix, (cols, rows))
    return image

# Random augment chooser
def apply_random_augmentation(image_bgr):
    library_choice = np.random.choice(['alb', 'imgaug', 'torch', 'keras', 'opencv'])

    if library_choice == 'alb':
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        aug = albumentations_transform(image=img_rgb)['image']
        return cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

    elif library_choice == 'imgaug':
        img = imageio.core.util.Array(image_bgr)
        aug = imgaug_seq(image=img)
        return aug

    elif library_choice == 'torch':
        img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        aug = torchvision_transform(img_pil)
        aug = transforms.ToPILImage()(transforms.ToTensor()(aug))
        return cv2.cvtColor(np.array(aug), cv2.COLOR_RGB2BGR)

    elif library_choice == 'keras':
        img = tf.convert_to_tensor(image_bgr, dtype=tf.float32)
        img = tf.image.resize(img, [256, 256])
        img = tf.expand_dims(img, 0)
        aug = keras_augment(img, training=True)
        aug = tf.squeeze(aug, axis=0).numpy().astype(np.uint8)
        return aug

    else:
        return opencv_augment(image_bgr)

# Main logic
def generate_augmented_images_from_flat_folder(src_dir, dest_dir, num_images=150):
    os.makedirs(dest_dir, exist_ok=True)

    all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print("No images found in source folder.")
        return

    for img_name in all_images:
        img_path = os.path.join(src_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        student_dest = os.path.join(dest_dir, base_name)
        os.makedirs(student_dest, exist_ok=True)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Couldn't read: {img_path}")
            continue

        print(f"Generating for {base_name}...")

        for i in range(num_images):
            aug_img = apply_random_augmentation(image)
            filename = os.path.join(student_dest, f"img_{i+1}.jpg")
            cv2.imwrite(filename, aug_img)

        print(f"    Saved 50 images for {base_name}")

    print("All augmentations complete.")

# ==== Set paths ====
source = "/Users/sivakarthick/Downloads/ML2_miniprj/Dataset/original_dataset_cropped"
target = "/Users/sivakarthick/Downloads/ML2_miniprj/Dataset/2023_27_AIML_augmented"

generate_augmented_images_from_flat_folder(source, target, num_images=50)
