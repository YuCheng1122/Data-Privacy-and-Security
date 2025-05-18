import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def blur_faces_in_image(image_path, net, confidence_threshold=0.5, blur_ksize=(151,151), blur_sigma=50):
    """Apply face blurring to a single image
    
    This function detects faces in an image using a pre-trained DNN model and applies
    Gaussian blur to each detected face region.
    
    Args:
        image_path (str): Path to the input image file
        net (cv2.dnn.Net): Pre-trained face detection model
        confidence_threshold (float, optional): Minimum confidence score to consider a detection valid. Defaults to 0.5.
        blur_ksize (tuple, optional): Gaussian blur kernel size. Defaults to (99, 99).
        blur_sigma (int, optional): Gaussian blur sigma value. Defaults to 30.
        
    Returns:
        tuple: (processed_image, face_found_flag)
            - processed_image: Image with blurred faces (or None if image loading failed)
            - face_found_flag: Boolean indicating whether any faces were detected
    """
    # Load the image from disk
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None, False
    
    original_image = image.copy()
    (h, w) = image.shape[:2]
    
    # Create a blob from the image and pass it through the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Track whether any faces were found
    face_found = False
    
    # Loop over all detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by confidence threshold
        if confidence > confidence_threshold:
            face_found = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure coordinates are within image boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Skip very small regions that might be false positives
            if (endX - startX) < 20 or (endY - startY) < 20:
                continue
                
            # Extract the face region and apply Gaussian blur
            face_region = image[startY:endY, startX:endX]
            blurred_face = cv2.GaussianBlur(face_region, blur_ksize, blur_sigma)
            image[startY:endY, startX:endX] = blurred_face
    
    return image, face_found

def verify_blur(original, blurred):
    """Verify that blurring was successfully applied
    
    This function compares the original and blurred images to determine if
    significant changes were made, indicating successful face blurring.
    
    Args:
        original (numpy.ndarray): The original unmodified image
        blurred (numpy.ndarray): The image after face blurring
        
    Returns:
        bool: True if blurring was successfully applied, False otherwise
    """
    if original is None or blurred is None:
        return False
    
    # Calculate the absolute difference between original and blurred images
    diff = cv2.absdiff(original, blurred)
    diff_sum = np.sum(diff)
    
    # Consider blurring successful if the difference exceeds a threshold
    return diff_sum > 100000