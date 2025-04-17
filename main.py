import cv2
from glob import glob
import numpy as np
import os

from pipeline.segment import segment_images
from pipeline.HSV import preprocess_images
from pipeline.normalization import align_images
from pipeline.ImgOutlier import detect_outliers

train_input = "./metalmarcy"
train_output = "./output"
model_path = "./model/best_model-epoch=11-valid_iou=0.9230.ckpt"


def load_images_from_path(path):
    """
    Load images from the given path, returning the first 5 alphabetically sorted images
    as reference images and the rest as training images.

    Args:
        path: String path to the directory containing images

    Returns:
        tuple: (reference_images, train_images) where each is a list of CV2-opened images
    """
    # Get all image files from the directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob(os.path.join(path, ext)))

    # Sort alphabetically
    image_files.sort()

    # Check if we have enough images
    if len(image_files) < 5:
        raise ValueError(f"Not enough images in {path}. Found {len(image_files)}, need at least 6.")

    # Split into reference and train sets
    reference_files = image_files[:4]
    train_files = image_files[4:]

    # Load images with OpenCV
    reference_images = []
    for file in reference_files:
        img = cv2.imread(file)
        if img is not None:
            reference_images.append(img)
        else:
            print(f"Warning: Could not load {file}")

    train_images = []
    for file in train_files:
        img = cv2.imread(file)
        if img is not None:
            train_images.append(img)
        else:
            print(f"Warning: Could not load {file}")

    return reference_images, train_images


def overlay_segmentation(images, segmentations, alpha=0.3):
    """
    Overlay segmentation images on top of original images with specified alpha transparency.

    Args:
        images: List of original images
        segmentations: List of segmentation images
        alpha: Transparency value for the overlay (0.0 to 1.0)

    Returns:
        List of overlaid images
    """
    if len(images) != len(segmentations):
        raise ValueError("Number of images and segmentations must match")

    overlaid_images = []

    for img, seg in zip(images, segmentations):
        # Make sure images have the same dimensions
        if img.shape[:2] != seg.shape[:2]:
            seg = cv2.resize(seg, (img.shape[1], img.shape[0]))

        # Create overlay
        overlaid = cv2.addWeighted(img, 1.0, seg, alpha, 0)
        overlaid_images.append(overlaid)

    return overlaid_images



# For whole dataset
ref_images, tra_images = load_images_from_path(train_input)
print("Done Loading")
ref_images = preprocess_images(ref_images)
tra_images = preprocess_images(tra_images)
print("Done Prepocessing")
filtered_images = detect_outliers(ref_images, tra_images)
print("Done removing outliers")
all_images = ref_images + filtered_images
all_segs = segment_images(all_images, model_path)
print("Done Segmenting")
output_images, output_segs = align_images(all_images, all_segs)
print("Done Normalizing")
final_images = overlay_segmentation(output_images, output_segs)
print("Done Overlaying")
paths = save_images_to_directory(final_images, train_output)



# For one Image
#ref_images, tra_images = load_images_from_path(train_input)
#print("Done Loading")
#ref_images = preprocess_images(ref_images)
#tra_images = preprocess_images(tra_images[0])
#print("Done Prepocessing")
#filtered_images = detect_outliers(ref_images, tra_images)
#print("Done removing outliers")
#all_images = ref_images[0] + filtered_images
#all_segs = segment_images(all_images, model_path)
#print("Done Segmenting")
#output_images, output_segs = align_images(all_images, all_segs)
#print("Done Normalizing")
#final_images = overlay_segmentation(output_images, output_segs)
#print("Done Overlaying")
#paths = save_images_to_directory(final_images, train_output)
