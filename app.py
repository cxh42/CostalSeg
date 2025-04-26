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
from PIL import Image
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
    try:
        ref_vector = np.load(vector_path)
        print(f"Reference vector loaded successfully: {vector_path}")
        return ref_vector
    except Exception as e:
        print(f"Reference vector loading failed {vector_path}: {e}")
        return []

# Load reference image
def load_reference_images(ref_dir):
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
    
    # Changed title to "Analysis Results"
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
    if image.shape[:2] != segmentation_map.shape[:2]:
        segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 1-alpha, segmentation_map, alpha, 0)

# Perform segmentation
def perform_segmentation(model, image_bgr):
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

# Single image processing
def process_coastal_image(location, input_image):
    if input_image is None:
        return None, None, "Please upload an image to analyze", "No image detected"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATHS[location], device)
    
    if model is None:
        return None, None, f"Error: Unable to load model", "Analysis failed"
    
    ref_vector = load_reference_vector(REFERENCE_VECTOR_PATHS[location]) if os.path.exists(REFERENCE_VECTOR_PATHS[location]) else []
    ref_images = load_reference_images(REFERENCE_IMAGE_DIRS[location])
    
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # Perform outlier detection
    is_outlier = False
    if len(ref_vector) > 0:
        filtered, _ = detect_outliers(ref_images, [image_bgr], ref_vector)
        is_outlier = len(filtered) == 0
    else:
        filtered, _ = detect_outliers(ref_images, [image_bgr])
        is_outlier = len(filtered) == 0
    
    outlier_status = "Outlier Detection: <span style='color:red;font-weight:bold'>Failed</span>" if is_outlier else "Outlier Detection: <span style='color:green;font-weight:bold'>Passed</span>"
    
    seg_map, overlay, analysis = perform_segmentation(model, image_bgr)
    
    if is_outlier:
        analysis = "<div style='color:red;font-weight:bold;margin-bottom:10px'>Warning: Image did not pass outlier detection. Results may be less accurate!</div>" + analysis
    
    return seg_map, overlay, analysis, outlier_status

# Spatial Alignment
def process_with_alignment(location, reference_image, input_image):
    if reference_image is None or input_image is None:
        return None, None, None, None, "Please upload both reference and target images for analysis", "Not processed"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATHS[location], device)
    
    if model is None:
        return None, None, None, None, "Error: Unable to load model", "Analysis failed"
    
    ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    aligned, _ = align_images([ref_bgr, tgt_bgr], [np.zeros_like(ref_bgr), np.zeros_like(tgt_bgr)])
    aligned_tgt_bgr = aligned[1]
    
    seg_map, overlay, analysis = perform_segmentation(model, aligned_tgt_bgr)
    
    status = "Spatial Alignment: <span style='color:green;font-weight:bold'>Successfully Completed</span>"
    
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    aligned_tgt_rgb = cv2.cvtColor(aligned_tgt_bgr, cv2.COLOR_BGR2RGB)
    
    return ref_rgb, aligned_tgt_rgb, seg_map, overlay, analysis, status

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Coastal Erosion Analysis System") as demo:
        gr.Markdown("""# Coastal Erosion Analysis System

Upload coastal photographs for segmentation analysis and spatial alignment. The system identifies terrain types including background, cobbles, sand, plants, sky, and water.""")
        
        with gr.Tabs():
            with gr.TabItem("Single Image Segmentation"):
                with gr.Row():
                    loc1 = gr.Radio(list(MODEL_PATHS.keys()), label="Select Location", value=list(MODEL_PATHS.keys())[0])
                
                with gr.Row():
                    inp = gr.Image(label="Input Image", type="numpy", image_mode="RGB")
                    seg = gr.Image(label="Segmentation Map", type="numpy")
                    ovl = gr.Image(label="Overlay Visualization", type="numpy")
                
                with gr.Row():
                    btn1 = gr.Button("Run Segmentation Analysis", variant="primary")
                
                status1 = gr.HTML(label="Outlier Detection Status")
                res1 = gr.HTML(label="Terrain Analysis")
                
                btn1.click(fn=process_coastal_image, inputs=[loc1, inp], outputs=[seg, ovl, res1, status1])
            
            with gr.TabItem("Spatial Alignment Segmentation"):
                with gr.Row():
                    loc2 = gr.Radio(list(MODEL_PATHS.keys()), label="Select Location", value=list(MODEL_PATHS.keys())[0])
                
                with gr.Row():
                    ref_img = gr.Image(label="Reference Image", type="numpy", image_mode="RGB")
                    tgt_img = gr.Image(label="Target Image for Analysis", type="numpy", image_mode="RGB")
                
                with gr.Row():
                    btn2 = gr.Button("Run Spatial Alignment Analysis", variant="primary")
                
                with gr.Row():
                    orig = gr.Image(label="Original Reference", type="numpy")
                    aligned = gr.Image(label="Aligned Image", type="numpy")
                
                with gr.Row():
                    seg2 = gr.Image(label="Segmentation Map", type="numpy")
                    ovl2 = gr.Image(label="Overlay Visualization", type="numpy")
                
                status2 = gr.HTML(label="Spatial Alignment Status")
                res2 = gr.HTML(label="Terrain Analysis")
                
                btn2.click(fn=process_with_alignment, inputs=[loc2, ref_img, tgt_img], outputs=[orig, aligned, seg2, ovl2, res2, status2])
    
    return demo

if __name__ == "__main__":
    for path in ["models", "reference_images/MM", "reference_images/SJ"]:
        os.makedirs(path, exist_ok=True)
    for p in MODEL_PATHS.values():
        if not os.path.exists(p):
            print(f"Error: Model file {p} does not exist!")
    demo = create_interface()
    # Added inbrowser=True to automatically open the browser
    demo.launch(inbrowser=True)