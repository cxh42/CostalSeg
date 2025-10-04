import os
import cv2
import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from glob import glob
from pipeline.ImgOutlier import detect_outliers
from pipeline.normalization import align_images

# Global Configuration
MODEL_PATHS = {
    "Metal Marcy": "models/MM_best_model.pth",
    "Silhouette Jaenette": "models/SJ_best_model.pth"
}

REFERENCE_VECTOR_PATHS = {
    "Metal Marcy": "models/MM_mean.npy",
    "Silhouette Jaenette": "models/SJ_mean.npy"
}

REFERENCE_IMAGE_DIRS = {
    "Metal Marcy": "reference_images/MM",
    "Silhouette Jaenette": "reference_images/SJ"
}

# Category names and color mapping
CLASSES = ['Background', 'Cobbles', 'Dry sand', 'Plant', 'Sky', 'Water', 'Wet sand']
COLORS = [
    [0, 0, 0],        # background - black
    [139, 137, 137],  # cobbles - dark gray
    [255, 228, 181],  # drysand - light yellow
    [0, 128, 0],      # plant - green
    [135, 206, 235],  # sky - sky blue
    [0, 0, 255],      # water - blue
    [194, 178, 128]   # wetsand - sand brown
]

# Load model function
def load_model(model_path, device="cuda"):
    """
    Load the segmentation model from the specified path
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded PyTorch model or None if loading failed
    """
    try:
        model = smp.create_model(
            "DeepLabV3Plus",
            encoder_name="efficientnet-b6",
            in_channels=3,
            classes=len(CLASSES),
            encoder_weights=None
        )
        state_dict = torch.load(model_path, map_location=device)
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

# Load reference vector
def load_reference_vector(vector_path):
    """
    Load the reference vector used for outlier detection
    
    Args:
        vector_path (str): Path to the reference vector file
        
    Returns:
        np.array: Reference vector or empty list if loading failed
    """
    try:
        ref_vector = np.load(vector_path)
        print(f"Reference vector loaded successfully: {vector_path}")
        return ref_vector
    except Exception as e:
        print(f"Reference vector loading failed {vector_path}: {e}")
        return []

# Load reference images
def load_reference_images(ref_dir):
    """
    Load reference images from the specified directory
    
    Args:
        ref_dir (str): Directory containing reference images
        
    Returns:
        list: List of loaded reference images or empty list if loading failed
    """
    try:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(ref_dir, ext)))
        image_files.sort()
        reference_images = []
        for file in image_files[:4]:
            img = cv2.imread(file)
            if img is not None:
                reference_images.append(img)
        print(f"Loaded {len(reference_images)} images from {ref_dir}")
        return reference_images
    except Exception as e:
        print(f"Image loading failed {ref_dir}: {e}")
        return []

# Preprocess the image
def preprocess_image(image):
    """
    Preprocess an image for model inference
    
    Args:
        image (np.array): Input image in RGB format
        
    Returns:
        tuple: (preprocessed image tensor, original height, original width)
    """
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (1024, 1024))
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return image_tensor, orig_h, orig_w

# Generate segmentation map and visualization
def generate_segmentation_map(prediction, orig_h, orig_w):
    """
    Generate a segmentation map from model prediction
    
    Args:
        prediction (torch.Tensor): Model prediction
        orig_h (int): Original image height
        orig_w (int): Original image width
        
    Returns:
        np.array: Segmentation map as a colored image
    """
    mask = prediction.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5, 5), np.uint8)
    processed_mask = mask_resized.copy()
    for idx in range(1, len(CLASSES)):
        class_mask = (mask_resized == idx).astype(np.uint8)
        dilated_mask = cv2.dilate(class_mask, kernel, iterations=2)
        dilated_effect = dilated_mask & (mask_resized == 0)
        processed_mask[dilated_effect > 0] = idx
    segmentation_map = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for idx, color in enumerate(COLORS):
        segmentation_map[processed_mask == idx] = color
    return segmentation_map

# Analysis result with Pie Chart (including background)
def create_analysis_result(mask):
    """
    Create a pie chart visualization of the terrain distribution
    
    Args:
        mask (np.array): Segmentation mask
        
    Returns:
        str: HTML content with embedded pie chart
    """
    # Calculate percentages for each class
    total_pixels = mask.size
    percentages = {cls: round((np.sum(mask == i) / total_pixels) * 100, 1)
                   for i, cls in enumerate(CLASSES) if np.sum(mask == i) > 0}
    
    # Sort percentages by value in descending order
    sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Create a Figure and canvas for the pie chart
    fig = Figure(figsize=(8, 5), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Create the pie chart with sorted data
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [np.array(COLORS[CLASSES.index(cls)])/255 for cls in labels]
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=None,  # We'll create a custom legend
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85
    )
    
    # Improve text legibility
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
        # Make text readable regardless of background color
        autotext.set_color('white' if np.mean(colors[autotexts.index(autotext)]) < 0.5 else 'black')
    
    # Create a legend with colored boxes in the same sorted order
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f"{labels[i]} ({values[i]}%)") 
                      for i in range(len(labels))]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    
    # Set title to "Analysis Results"
    ax.set_title('Analysis Results')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Convert the plot to an image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
    
    # Create the HTML content with the embedded image
    result = f"""
    <div style='display:flex; flex-direction:column; align-items:center;'>
        <img src='{img_str}' alt='Terrain Distribution Pie Chart' style='max-width:100%; height:auto;'>
    </div>
    """
    
    return result

# Merge and overlay
def create_overlay(image, segmentation_map, alpha=0.5):
    """
    Create an overlay of the original image and segmentation map
    
    Args:
        image (np.array): Original image in RGB format
        segmentation_map (np.array): Segmentation map
        alpha (float): Transparency value for the overlay
        
    Returns:
        np.array: Overlay image
    """
    if image.shape[:2] != segmentation_map.shape[:2]:
        segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 1-alpha, segmentation_map, alpha, 0)

# Perform segmentation
def perform_segmentation(model, image_bgr):
    """
    Perform segmentation on an image
    
    Args:
        model: Loaded PyTorch model
        image_bgr (np.array): Input image in BGR format
        
    Returns:
        tuple: (segmentation map, overlay image, analysis HTML)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor, orig_h, orig_w = preprocess_image(image_rgb)
    with torch.no_grad():
        prediction = model(image_tensor.to(device))
    seg_map = generate_segmentation_map(prediction, orig_h, orig_w)  # RGB
    overlay = create_overlay(image_rgb, seg_map)
    mask = prediction.argmax(1).squeeze().cpu().numpy()
    analysis = create_analysis_result(mask)
    return seg_map, overlay, analysis

def predict_mask_and_analysis(model, image_bgr):
    """
    Run model on the original (pre-alignment) image and return
    the raw class-index mask (model resolution) and analysis HTML.

    Args:
        model: Loaded PyTorch model
        image_bgr (np.array): Input image in BGR format

    Returns:
        tuple: (mask_1024, orig_h, orig_w, analysis_html)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor, orig_h, orig_w = preprocess_image(image_rgb)
    with torch.no_grad():
        prediction = model(image_tensor.to(device))
    mask_1024 = prediction.argmax(1).squeeze().cpu().numpy()
    analysis = create_analysis_result(mask_1024)
    return mask_1024, orig_h, orig_w, analysis

# Split the processing into separate functions for progressive display

def run_segmentation(location, input_image, progress=gr.Progress()):
    """
    Run image segmentation task independently
    
    Args:
        location (str): Location name for model selection
        input_image (np.array): Input image
        progress: Gradio progress indicator
        
    Returns:
        tuple: (segmentation map, overlay image, analysis HTML)
    """
    if input_image is None:
        return None, None, "Please upload an image to analyze"
    
    # Set up GPU device
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading segmentation model...")
    model = load_model(MODEL_PATHS[location], gpu_device)
    
    if model is None:
        return None, None, "Error: Unable to load model"
    
    # Process the image
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    progress(0.3, desc="Performing segmentation (GPU)...")
    seg_map, overlay, analysis = perform_segmentation(model, image_bgr)
    
    progress(1.0, desc="Segmentation complete")
    return seg_map, overlay, analysis

def run_outlier_detection(location, input_image, progress=gr.Progress()):
    """
    Run outlier detection task independently
    
    Args:
        location (str): Location name for model selection
        input_image (np.array): Input image
        progress: Gradio progress indicator
        
    Returns:
        str: Outlier detection status HTML
    """
    if input_image is None:
        return "No image detected"
    
    # Choose device for outlier detection (prefer GPU if available)
    det_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading reference data...")
    
    # Load reference data
    ref_vector = load_reference_vector(REFERENCE_VECTOR_PATHS[location]) if os.path.exists(REFERENCE_VECTOR_PATHS[location]) else []
    ref_images = load_reference_images(REFERENCE_IMAGE_DIRS[location])
    
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # Perform outlier detection
    progress(0.3, desc=f"Performing outlier detection ({det_device.upper()})...")
    is_outlier = False
    
    # Run detection using selected device
    if len(ref_vector) > 0:
        filtered, _ = detect_outliers(ref_images, [image_bgr], ref_vector, device=det_device)
        is_outlier = len(filtered) == 0
    else:
        filtered, _ = detect_outliers(ref_images, [image_bgr], device=det_device)
        is_outlier = len(filtered) == 0
    
    progress(1.0, desc="Outlier detection complete")
    outlier_status = "Outlier Detection: <span style='color:red;font-weight:bold'>Failed</span>" if is_outlier else "Outlier Detection: <span style='color:green;font-weight:bold'>Passed</span>"
    
    # Add warning to analysis if outlier
    if is_outlier:
        outlier_warning = "<div style='color:red;font-weight:bold;margin-bottom:10px'>Warning: Image did not pass outlier detection. Results may be less accurate!</div>"
        return outlier_status, outlier_warning
    
    return outlier_status, ""

def update_analysis_with_warning(analysis, warning):
    """
    Update analysis HTML with warning message if needed
    
    Args:
        analysis (str): Original analysis HTML
        warning (str): Warning message to prepend
        
    Returns:
        str: Updated analysis HTML
    """
    if warning and analysis:
        return warning + analysis
    return analysis

# Spatial Alignment with progressive display
def run_alignment_and_segmentation(location, reference_image, input_image, progress=gr.Progress()):
    """
    Run spatial alignment and segmentation with progressive display
    
    Args:
        location (str): Location name for model selection
        reference_image (np.array): Reference image
        input_image (np.array): Input image to analyze
        progress: Gradio progress indicator
        
    Returns:
        tuple: (
            reference image,
            aligned image,
            segmentation map (aligned),
            overlay image (aligned),
            pre-alignment analysis HTML,
            aligned analysis HTML,
            status HTML
        )
    """
    if reference_image is None or input_image is None:
        return None, None, None, None, "Please upload both reference and target images for analysis", "Not processed"
    
    # Set up GPU device
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading segmentation model...")
    
    model = load_model(MODEL_PATHS[location], gpu_device)
    
    if model is None:
        return None, None, None, None, "Error: Unable to load model", "Analysis failed"
    
    ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # 1) Perform segmentation on pre-aligned target image
    progress(0.3, desc="Performing segmentation (pre-alignment)...")
    mask_1024, orig_h, orig_w, analysis_pre = predict_mask_and_analysis(model, tgt_bgr)

    # Resize mask back to original target size and lightly postprocess (same as visualization path)
    mask_resized = cv2.resize(mask_1024, (tgt_bgr.shape[1], tgt_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5, 5), np.uint8)
    processed_mask = mask_resized.copy()
    for idx in range(1, len(CLASSES)):
        class_mask = (mask_resized == idx).astype(np.uint8)
        dilated_mask = cv2.dilate(class_mask, kernel, iterations=2)
        dilated_effect = dilated_mask & (mask_resized == 0)
        processed_mask[dilated_effect > 0] = idx

    # 2) Perform spatial alignment on images and warp the segmentation mask using the same transform
    progress(0.6, desc="Performing spatial alignment...")
    ref_seg_dummy = np.zeros(ref_bgr.shape[:2], dtype=np.uint8)
    aligned_imgs, aligned_segs = align_images([ref_bgr, tgt_bgr], [ref_seg_dummy, processed_mask.astype(np.uint8)])
    aligned_tgt_bgr = aligned_imgs[1]
    aligned_mask = aligned_segs[1]

    # 3) Prepare display images
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    aligned_tgt_rgb = cv2.cvtColor(aligned_tgt_bgr, cv2.COLOR_BGR2RGB)

    # Colorize aligned mask for visualization
    seg_map_aligned = np.zeros((aligned_mask.shape[0], aligned_mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(COLORS):
        seg_map_aligned[aligned_mask == idx] = color

    # Overlay on aligned image
    overlay_aligned = create_overlay(aligned_tgt_rgb, seg_map_aligned)

    status = "Spatial Alignment: <span style='color:green;font-weight:bold'>Successfully Completed</span>"

    # 4) Compute analysis from aligned mask (includes black borders)
    analysis_aligned = create_analysis_result(aligned_mask.astype(np.uint8))

    # 5) Return both analyses: pre-alignment and aligned
    progress(1.0, desc="Analysis complete")
    return ref_rgb, aligned_tgt_rgb, seg_map_aligned, overlay_aligned, analysis_pre, analysis_aligned, status

# Create the Gradio interface with progressive display
def create_interface():
    """
    Create the Gradio web interface with progressive result display
    
    Returns:
        gradio.Blocks: Gradio interface
    """
    with gr.Blocks(title="Coastal Erosion Analysis System") as demo:
        gr.Markdown("""# Coastal Erosion Analysis System

Upload coastal photographs for segmentation analysis and spatial alignment. The system identifies terrain types including background, cobbles, sand, plants, sky, and water.""")
        
        # Store analysis content for updating with warnings
        current_analysis = gr.State("")
        outlier_warning = gr.State("")
        
        with gr.Tabs():
            with gr.TabItem("Single Image Segmentation"):
                with gr.Row():
                    loc1 = gr.Radio(list(MODEL_PATHS.keys()), label="Select Location", value=list(MODEL_PATHS.keys())[0])

                with gr.Row():
                    inp = gr.Image(label="Input Image", type="numpy", image_mode="RGB")
                    seg = gr.Image(label="Segmentation Map", type="numpy")
                    ovl = gr.Image(label="Overlay Visualization", type="numpy")

                with gr.Row():
                    btn1 = gr.Button("Run Analysis", variant="primary")

                # Built-in example images
                gr.Examples(
                    label="Examples",
                    examples=[
                        ["Metal Marcy", "reference_images/MM/2025-01-26_16-36-00_MM.jpg"],
                        ["Metal Marcy", "reference_images/MM/2025-01-25_13-55-00_MM.jpg"],
                        ["Silhouette Jaenette", "reference_images/SJ/2025-01-26_14-43-00_SJ.jpg"],
                        ["Silhouette Jaenette", "reference_images/SJ/2025-01-23_11-22-00_SJ.jpg"],
                    ],
                    inputs=[loc1, inp],
                    examples_per_page=4,
                )

                status1 = gr.HTML(label="Outlier Detection Status")
                res1 = gr.HTML(label="Terrain Analysis")
                
                # When the button is clicked, run both functions in parallel
                btn1.click(
                    fn=run_segmentation,
                    inputs=[loc1, inp],
                    outputs=[seg, ovl, res1]
                ).then(
                    fn=lambda analysis: analysis,
                    inputs=[res1],
                    outputs=[current_analysis]
                )
                
                # Also start outlier detection
                btn1.click(
                    fn=run_outlier_detection,
                    inputs=[loc1, inp],
                    outputs=[status1, outlier_warning]
                ).then(
                    # Update analysis with warning if needed
                    fn=update_analysis_with_warning,
                    inputs=[current_analysis, outlier_warning],
                    outputs=[res1]
                )
            
            with gr.TabItem("Spatial Alignment Segmentation"):
                with gr.Row():
                    loc2 = gr.Radio(list(MODEL_PATHS.keys()), label="Select Location", value=list(MODEL_PATHS.keys())[0])

                with gr.Row():
                    ref_img = gr.Image(label="Reference Image", type="numpy", image_mode="RGB")
                    tgt_img = gr.Image(label="Target Image for Analysis", type="numpy", image_mode="RGB")

                with gr.Row():
                    btn2 = gr.Button("Run Spatial Alignment Analysis", variant="primary")

                # Built-in paired examples (reference, target)
                gr.Examples(
                    label="Examples",
                    examples=[
                        [
                            "Metal Marcy",
                            "reference_images/MM/2025-01-26_16-36-00_MM.jpg",
                            "reference_images/MM/2025-01-25_13-55-00_MM.jpg",
                        ],
                        [
                            "Silhouette Jaenette",
                            "reference_images/SJ/2025-01-26_14-43-00_SJ.jpg",
                            "reference_images/SJ/2025-01-23_11-22-00_SJ.jpg",
                        ],
                    ],
                    inputs=[loc2, ref_img, tgt_img],
                    examples_per_page=2,
                )

                with gr.Row():
                    orig = gr.Image(label="Original Reference", type="numpy")
                    aligned = gr.Image(label="Aligned Image", type="numpy")
                
                with gr.Row():
                    seg2 = gr.Image(label="Segmentation Map", type="numpy")
                    ovl2 = gr.Image(label="Overlay Visualization", type="numpy")
                
                status2 = gr.HTML(label="Spatial Alignment Status")
                res2_pre = gr.HTML(label="Pre-alignment Analysis")
                res2_aligned = gr.HTML(label="Aligned Analysis (with borders)")
                
                # For alignment, we use the progressive display function
                btn2.click(
                    fn=run_alignment_and_segmentation,
                    inputs=[loc2, ref_img, tgt_img],
                    outputs=[orig, aligned, seg2, ovl2, res2_pre, res2_aligned, status2]
                )
    
    return demo

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    for path in ["models", "reference_images/MM", "reference_images/SJ"]:
        os.makedirs(path, exist_ok=True)
    
    # Check if model files exist
    for p in MODEL_PATHS.values():
        if not os.path.exists(p):
            print(f"Error: Model file {p} does not exist!")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(inbrowser=True)
