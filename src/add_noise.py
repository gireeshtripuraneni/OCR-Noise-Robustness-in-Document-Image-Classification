import os
import cv2
import numpy as np

# =====================================
# Configuration
# =====================================

IMAGE_PATH = "data/form/00040534.tif"

OUTPUT_DIR = "results/noisy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# Read Image
# =====================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# =====================================
# Gaussian Noise
# =====================================

def gaussian_noise(img, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# =====================================
# Salt & Pepper Noise
# =====================================

def salt_pepper_noise(img, amount=0.02):

    noisy = img.copy()

    num_salt = int(amount * img.size * 0.5)

    coords = [
        np.random.randint(0, i - 1, num_salt)
        for i in img.shape[:2]
    ]

    noisy[coords[0], coords[1]] = 255

    num_pepper = int(amount * img.size * 0.5)

    coords = [
        np.random.randint(0, i - 1, num_pepper)
        for i in img.shape[:2]
    ]

    noisy[coords[0], coords[1]] = 0

    return noisy

# =====================================
# Gaussian Blur
# =====================================

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# =====================================
# Motion Blur
# =====================================

def motion_blur(img, size=15):

    kernel = np.zeros((size, size))

    kernel[int((size - 1) / 2), :] = np.ones(size)

    kernel = kernel / size

    return cv2.filter2D(img, -1, kernel)

# =====================================
# Rotation
# =====================================

def rotate(img, angle):

    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(img, matrix, (w, h))

# =====================================
# Save Images
# =====================================

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "original.png"),
    image
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "gaussian_noise.png"),
    gaussian_noise(image)
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "salt_pepper.png"),
    salt_pepper_noise(image)
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "gaussian_blur.png"),
    gaussian_blur(image)
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "motion_blur.png"),
    motion_blur(image)
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "rotation_5.png"),
    rotate(image, 5)
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "rotation_10.png"),
    rotate(image, 10)
)

print("Noisy images saved successfully!")
print(f"Location: {OUTPUT_DIR}")