import os
import cv2
import numpy as np
import torch
from glob import glob
from pipeline.ImgOutlier import CosineSimilarity
from pipeline.HSV import preprocess_images

def create_reference_vector(image_dir, output_path, threshold=0.8):
    """
    Create a reference vector from a set of representative images
    
    Parameters:
        image_dir: Directory containing representative images
        output_path: Path to save the reference vector
        threshold: Outlier detection threshold
    """
    print(f"Loading images from {image_dir}...")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
    
    # Sort image files
    image_files.sort()
    
    if len(image_files) < 10:
        print(f"Warning: Only found {len(image_files)} images. It is recommended to have at least 20 images for better reference vectors.")
    
    # Load images
    ref_images = []
    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            ref_images.append(img)
        else:
            print(f"Warning: Unable to load {file}")
    
    print(f"Successfully loaded {len(ref_images)} images")
    
    # Preprocess images
    print("Preprocessing images...")
    preprocessed_images = preprocess_images(ref_images)
    
    # Create cosine similarity instance
    similarity = CosineSimilarity(vector='feature', threshold=threshold)
    
    # Get reference vector - fix this line
    print("Extracting features and computing reference vector...")
    # find_outliers returns three values: mask, scores, emb_ref
    mask, scores, mean_vector = similarity.find_outliers(preprocessed_images, preprocessed_images)
    
    # Save reference vector
    print(f"Saving reference vector to {output_path}")
    np.save(output_path, mean_vector)
    print(f"Reference vector saved, shape: {mean_vector.shape}")
    return mean_vector

if __name__ == "__main__":
    # Metal Marcy reference vector
    mm_ref_dir = "reference_images/MM"
    mm_output = "models/MM_mean.npy"
    
    # Silhouette Jaenette reference vector
    sj_ref_dir = "reference_images/SJ" 
    sj_output = "models/SJ_mean.npy"
    
    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)
    
    # Choose which reference vector to generate
    location = input("Generate reference vector for which location? (1: Metal Marcy, 2: Silhouette Jaenette): ")
    
    if location == "1":
        vector = create_reference_vector(mm_ref_dir, mm_output)
        print("Metal Marcy reference vector generated")
    elif location == "2":
        vector = create_reference_vector(sj_ref_dir, sj_output)
        print("Silhouette Jaenette reference vector generated")
    else:
        print("Invalid selection")
