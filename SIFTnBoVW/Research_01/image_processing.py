import cv2
import numpy as np
import glob
import os


def load_images(image_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir,'*.jpg')))
    bgr_images = []
    gray_images = []
    for image_path in image_paths:
        bgr = cv2.imread(image_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr_images.append(bgr)
        gray_images.append(gray)
    return bgr_images, gray_images


def extract_visual_features(gray_images):
# Extract SIFT features from gray images
    # Define our feature extractor (SIFT)
    extractor = cv2.SIFT_create()
    
    keypoints = []
    descriptors = []

    for img in gray_images:
        # extract keypoints and descriptors for each image
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        if img_descriptors is not None:
            keypoints.append(img_keypoints)
            descriptors.append(img_descriptors)
    return keypoints, descriptors


def visualize_keypoints(bgr_image, image_keypoints):
    cv2.drawKeypoints(bgr_image, image_keypoints, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return bgr_image.copy()

