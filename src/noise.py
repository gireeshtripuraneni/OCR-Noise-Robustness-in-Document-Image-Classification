import cv2
import numpy as np

# ======================================================
# Gaussian Noise
# ======================================================

def gaussian_noise(image, mean=0, sigma=20):
    """
    Add Gaussian Noise
    """

    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


# ======================================================
# Salt & Pepper Noise
# ======================================================

def salt_pepper_noise(image, amount=0.02):
    """
    Add Salt & Pepper Noise
    """

    noisy = image.copy()

    h, w = noisy.shape[:2]

    num_pixels = int(amount * h * w)

    # Salt

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    noisy[ys, xs] = 255

    # Pepper

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    noisy[ys, xs] = 0

    return noisy


# ======================================================
# Gaussian Blur
# ======================================================

def gaussian_blur(image, kernel_size=5):

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


# ======================================================
# Motion Blur
# ======================================================

def motion_blur(image, kernel_size=15):

    kernel = np.zeros((kernel_size, kernel_size))

    kernel[kernel_size // 2, :] = np.ones(kernel_size)

    kernel /= kernel_size

    return cv2.filter2D(image, -1, kernel)


# ======================================================
# Rotation
# ======================================================

def rotate(image, angle):

    h, w = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )


# ======================================================
# Brightness
# ======================================================

def brightness(image, value=40):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv = hsv.astype(np.int16)

    hsv[:, :, 2] += value

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2],
        0,
        255
    )

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


# ======================================================
# Contrast
# ======================================================

def contrast(image, alpha=1.4):

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=0
    )


# ======================================================
# Noise Dictionary
# ======================================================

NOISE_FUNCTIONS = {

    "original":
        lambda x: x,

    "gaussian_noise":
        gaussian_noise,

    "salt_pepper":
        salt_pepper_noise,

    "gaussian_blur":
        gaussian_blur,

    "motion_blur":
        motion_blur,

    "rotation_5":
        lambda x: rotate(x, 5),

    "rotation_10":
        lambda x: rotate(x, 10),

    "brightness":
        brightness,

    "contrast":
        contrast
}