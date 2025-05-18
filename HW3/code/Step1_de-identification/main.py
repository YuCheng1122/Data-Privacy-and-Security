import os
import cv2
from pathlib import Path
from blur import blur_faces_in_image, verify_blur
from tqdm import tqdm
from matplotlib import pyplot as plt

def process_dataset(data_path, output_path, proto_path, model_path):
    """Process all images in the dataset
    
    This function iterates through all images in the specified directory,
    applies face blurring to each image, and saves them to the output directory.
    Failed cases are stored in a separate subfolder.
    
    Args:
        data_path (str): Path to the directory containing the original images
        output_path (str): Path to the directory where processed images will be saved
        proto_path (str): Path to the .prototxt file defining the model architecture
        model_path (str): Path to the pre-trained Caffe model weights
        
    Returns:
        dict: Statistics about the processing results including:
            - Total number of images
            - Successfully blurred images
            - Images with no faces detected
            - Images where blurring failed to apply correctly
    """
    # Load the face detection model
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    # Ensure output directories exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "failed"), exist_ok=True)
    
    # Get all image files from the data directory
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_files = []
    
    for ext in image_extensions:
        all_files.extend(list(Path(data_path).glob(f"**/*{ext}")))
    
    print(f"Found {len(all_files)} images")
    
    # Initialize statistics counters
    successful = 0
    failed_detection = 0
    failed_blur = 0
    
    # Process each image with a progress bar
    for img_path in tqdm(all_files):
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(output_path, img_name)
        
        # Read the original image
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
            
        # Apply face blurring to the image
        blurred_img, face_found = blur_faces_in_image(str(img_path), net)
        
        if not face_found:
            # No face detected, save to the failed directory
            cv2.imwrite(os.path.join(output_path, "failed", img_name), original)
            failed_detection += 1
            continue
        
        # Verify that the blur was successfully applied
        if verify_blur(original, blurred_img):
            # Save the successfully blurred image
            cv2.imwrite(output_img_path, blurred_img)
            successful += 1
        else:
            # Blur verification failed, save to the failed directory
            cv2.imwrite(os.path.join(output_path, "failed", img_name), original)
            failed_blur += 1
    
    # Return processing statistics
    return {
        "Total images": len(all_files),
        "Successfully blurred": successful,
        "No face detected": failed_detection,
        "Blur processing failed": failed_blur
    }

def visualize_results(stats):
    """Visualize the processing statistics
    
    Creates a bar chart showing the distribution of processing outcomes:
    successful blurs, failed face detections, and failed blur applications.
    
    Args:
        stats (dict): Statistics dictionary returned by process_dataset()
    
    Returns:
        None: Displays the visualization using matplotlib
    """
    labels = list(stats.keys())[1:]  # Exclude the total image count
    values = [stats[key] for key in labels]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['green', 'red', 'orange'])
    plt.title('Face Blurring Results')
    plt.ylabel('Number of Images')
    plt.show()
    
if __name__ == "__main__":
    data_path = "/home/tommy/.cache/kagglehub/datasets/lamsimon/celebahq/versions/1"
    output_path = "/home/tommy/Homework/DataPrivacy/HW3/blurred_images_50"
    proto_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    stats = process_dataset(data_path, output_path, proto_path, model_path)
    print("\nResults:")
    for key, value in stats.items():
        print(f"{key}: {value}")