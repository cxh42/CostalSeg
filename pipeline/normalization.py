from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
import cv2


def align_images(images, segs):
    """
    Align images using SuperGlue for feature matching.

    Args:
        images: List of input images
        segs: List of segmentation images

    Returns:
        Tuple of (aligned images, aligned segmentation images)
    """
    if not images or len(images) < 2:
        return images, segs

    reference = images[0]
    reference_seg = segs[0]
    aligned_images = [reference]
    aligned_images_seg = [reference_seg]

    # Load SuperGlue model and processor
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
    model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

    for i in range(1, len(images)):
        current = images[i]
        current_seg = segs[i]

        # Process image pair
        image_pair = [reference, current]
        inputs = processor(image_pair, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Get matches
        image_sizes = [[(img.shape[0], img.shape[1]) for img in image_pair]]
        matches = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

        # Extract matching keypoints
        match_data = matches[0]
        keypoints0 = match_data["keypoints0"].numpy()
        keypoints1 = match_data["keypoints1"].numpy()

        # Filter matches by confidence
        valid_matches = match_data["matching_scores"] > 0.5
        if sum(valid_matches) < 4:
            print(f"Not enough confident matches for image {i}, keeping original")
            aligned_images.append(current)
            aligned_images_seg.append(current_seg)
            continue

        # Get matching points
        src_pts = keypoints1[valid_matches].reshape(-1, 1, 2)
        dst_pts = keypoints0[valid_matches].reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # Apply homography
            h, w = reference.shape[:2]
            aligned = cv2.warpPerspective(current, H, (w, h))
            aligned_images.append(aligned)
            aligned_seg = cv2.warpPerspective(current_seg, H, (w, h))
            aligned_images_seg.append(aligned_seg)
        else:
            print(f"Could not find homography for image {i}, keeping original")
            aligned_images.append(current)
            aligned_images_seg.append(current_seg)

    return aligned_images, aligned_images_seg
