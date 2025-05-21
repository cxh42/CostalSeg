import os
import cv2
import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import io
import base64
import time
import psutil
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from glob import glob
from pipeline.ImgOutlier import detect_outliers
from pipeline.normalization import align_images

# 性能监控信息类
class PerformanceMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.start_mem_cpu = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_mem_gpu = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        else:
            self.start_mem_gpu = 0
            
    def start_stage(self, stage_name):
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
    def end_stage(self):
        stage_time = time.time() - self.stage_start_time
        self.stage_times[self.current_stage] = stage_time
        return stage_time
        
    def get_total_time(self):
        return time.time() - self.start_time
        
    def get_memory_usage(self):
        current_mem_cpu = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        cpu_mem_used = current_mem_cpu - self.start_mem_cpu
        
        if torch.cuda.is_available():
            current_mem_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            gpu_mem_used = current_mem_gpu - self.start_mem_gpu
        else:
            gpu_mem_used = 0
            
        return cpu_mem_used, gpu_mem_used
        
    def get_performance_report(self, image_size):
        total_time = self.get_total_time()
        cpu_mem, gpu_mem = self.get_memory_usage()
        
        # 计算每秒处理的像素数 (MPixels/s)
        pixels = image_size[0] * image_size[1]
        pixels_per_second = pixels / total_time / 1_000_000  # MPixels/s
        
        report = f"""
        <div style='background-color:#f0f7ff;padding:15px;border-radius:10px;margin:10px 0;border:1px solid #cce5ff'>
            <h3 style='color:#0066cc;margin-top:0'>性能监控报告</h3>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>
                <div>
                    <h4>时间统计</h4>
                    <table style='width:100%;border-collapse:collapse'>
                        <tr><td>总运行时间:</td><td style='text-align:right'><b>{total_time:.3f} 秒</b></td></tr>
        """
        
        # 添加各阶段时间
        for stage, t in self.stage_times.items():
            percentage = (t / total_time) * 100
            report += f"<tr><td>{stage}:</td><td style='text-align:right'>{t:.3f} 秒 ({percentage:.1f}%)</td></tr>"
            
        # 添加资源使用情况
        report += f"""
                    </table>
                </div>
                <div>
                    <h4>资源使用</h4>
                    <table style='width:100%;border-collapse:collapse'>
                        <tr><td>CPU内存使用:</td><td style='text-align:right'><b>{cpu_mem:.2f} MB</b></td></tr>
                        <tr><td>GPU内存使用:</td><td style='text-align:right'><b>{gpu_mem:.2f} MB</b></td></tr>
                        <tr><td>处理速度:</td><td style='text-align:right'><b>{pixels_per_second:.2f} MPixels/s</b></td></tr>
                        <tr><td>图像尺寸:</td><td style='text-align:right'>{image_size[0]}x{image_size[1]} px</td></tr>
                    </table>
                </div>
            </div>
        </div>
        """
        return report

# 创建全局性能监控器实例
performance_monitor = PerformanceMonitor()

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
    performance_monitor.start_stage("模型加载")
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
        performance_monitor.end_stage()
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        performance_monitor.end_stage()
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
    performance_monitor.start_stage("加载参考向量")
    try:
        ref_vector = np.load(vector_path)
        print(f"Reference vector loaded successfully: {vector_path}")
        performance_monitor.end_stage()
        return ref_vector
    except Exception as e:
        print(f"Reference vector loading failed {vector_path}: {e}")
        performance_monitor.end_stage()
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
    performance_monitor.start_stage("加载参考图像")
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
        performance_monitor.end_stage()
        return reference_images
    except Exception as e:
        print(f"Image loading failed {ref_dir}: {e}")
        performance_monitor.end_stage()
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
    performance_monitor.start_stage("图像预处理")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (1024, 1024))
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    performance_monitor.end_stage()
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
    performance_monitor.start_stage("生成分割图")
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
    performance_monitor.end_stage()
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
    performance_monitor.start_stage("生成分析结果")
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
    
    performance_monitor.end_stage()
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
    performance_monitor.start_stage("创建叠加图")
    if image.shape[:2] != segmentation_map.shape[:2]:
        segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = cv2.addWeighted(image, 1-alpha, segmentation_map, alpha, 0)
    performance_monitor.end_stage()
    return result

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
    performance_monitor.start_stage("模型推理")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor, orig_h, orig_w = preprocess_image(image_rgb)
    with torch.no_grad():
        prediction = model(image_tensor.to(device))
    performance_monitor.end_stage()
    
    seg_map = generate_segmentation_map(prediction, orig_h, orig_w)  # RGB
    overlay = create_overlay(image_rgb, seg_map)
    mask = prediction.argmax(1).squeeze().cpu().numpy()
    analysis = create_analysis_result(mask)
    
    # 添加性能报告到分析结果
    perf_report = performance_monitor.get_performance_report((orig_h, orig_w))
    analysis += perf_report
    
    return seg_map, overlay, analysis

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
    # 重置性能监控器
    performance_monitor.reset()
    performance_monitor.start_stage("总体流程")
    
    if input_image is None:
        performance_monitor.end_stage()
        return None, None, "Please upload an image to analyze"
    
    # 记录图像尺寸
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    image_h, image_w = image_bgr.shape[:2]
    
    # Set up GPU device
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading segmentation model...")
    model = load_model(MODEL_PATHS[location], gpu_device)
    
    if model is None:
        performance_monitor.end_stage()
        return None, None, "Error: Unable to load model"
    
    # Process the image
    progress(0.3, desc=f"Performing segmentation on {gpu_device.upper()}...")
    seg_map, overlay, analysis = perform_segmentation(model, image_bgr)
    
    # 结束总体计时
    performance_monitor.end_stage()
    
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
    performance_monitor.start_stage("异常检测")
    
    if input_image is None:
        performance_monitor.end_stage()
        return "No image detected"
    
    # Set up CPU device
    cpu_device = "cpu"
    
    # Show loading status
    progress(0, desc="Loading reference data...")
    
    # Load reference data
    ref_vector = load_reference_vector(REFERENCE_VECTOR_PATHS[location]) if os.path.exists(REFERENCE_VECTOR_PATHS[location]) else []
    ref_images = load_reference_images(REFERENCE_IMAGE_DIRS[location])
    
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # Perform outlier detection
    progress(0.3, desc="Performing outlier detection (CPU)...")
    is_outlier = False
    
    outlier_start = time.time()
    # Force CPU usage for outlier detection
    with torch.device(cpu_device):
        if len(ref_vector) > 0:
            filtered, _ = detect_outliers(ref_images, [image_bgr], ref_vector)
            is_outlier = len(filtered) == 0
        else:
            filtered, _ = detect_outliers(ref_images, [image_bgr])
            is_outlier = len(filtered) == 0
    outlier_time = time.time() - outlier_start
    
    performance_monitor.end_stage()
    progress(1.0, desc="Outlier detection complete")
    
    outlier_status = f"<div style='padding:10px;border-radius:5px;margin-top:10px'>"
    outlier_status += f"异常检测耗时: {outlier_time:.3f} 秒<br>"
    outlier_status += "检测状态: "
    
    if is_outlier:
        outlier_status += "<span style='color:red;font-weight:bold'>失败</span>"
        outlier_warning = "<div style='color:red;font-weight:bold;margin-bottom:10px'>警告：图像未通过异常检测。结果可能不太准确！</div>"
    else:
        outlier_status += "<span style='color:green;font-weight:bold'>通过</span>"
        outlier_warning = ""
        
    outlier_status += "</div>"
    
    return outlier_status, outlier_warning

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
        tuple: (reference image, aligned image, segmentation map, overlay image, analysis HTML, status HTML)
    """
    # 重置性能监控器
    performance_monitor.reset()
    performance_monitor.start_stage("总体流程")
    
    if reference_image is None or input_image is None:
        performance_monitor.end_stage()
        return None, None, None, None, "Please upload both reference and target images for analysis", "Not processed"
    
    # Set up GPU device
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading segmentation model...")
    
    model = load_model(MODEL_PATHS[location], gpu_device)
    
    if model is None:
        performance_monitor.end_stage()
        return None, None, None, None, "Error: Unable to load model", "Analysis failed"
    
    ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # Perform alignment (this step is needed before segmentation)
    progress(0.3, desc="Performing spatial alignment...")
    performance_monitor.start_stage("空间对齐")
    aligned, _ = align_images([ref_bgr, tgt_bgr], [np.zeros_like(ref_bgr), np.zeros_like(tgt_bgr)])
    performance_monitor.end_stage()
    
    aligned_tgt_bgr = aligned[1]
    
    # Convert images for display
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    aligned_tgt_rgb = cv2.cvtColor(aligned_tgt_bgr, cv2.COLOR_BGR2RGB)
    
    # Show alignment complete status
    progress(0.5, desc="Alignment complete, performing segmentation...")
    
    # Now perform segmentation
    seg_map, overlay, analysis = perform_segmentation(model, aligned_tgt_bgr)
    
    # 记录图像尺寸
    image_h, image_w = aligned_tgt_bgr.shape[:2]
    
    # 结束总体计时
    total_time = performance_monitor.end_stage()
    
    # 添加对齐性能信息
    status = f"""
    <div style='background-color:#f0fff0;padding:15px;border-radius:10px;margin:10px 0;border:1px solid #ccffcc'>
        <h3 style='color:#006600;margin-top:0'>空间对齐状态</h3>
        <p>状态: <span style='color:green;font-weight:bold'>成功完成</span></p>
        <p>对齐耗时: {performance_monitor.stage_times.get('空间对齐', 0):.3f} 秒</p>
        <p>图像尺寸: {image_w}x{image_h} 像素</p>
    </div>
    """
    
    progress(1.0, desc="Analysis complete")
    return ref_rgb, aligned_tgt_rgb, seg_map, overlay, analysis, status

# Create the Gradio interface with progressive display
def create_interface():
    """
    Create the Gradio web interface with progressive result display
    
    Returns:
        gradio.Blocks: Gradio interface
    """
    with gr.Blocks(title="Coastal Erosion Analysis System") as demo:
        gr.Markdown("""# 海岸侵蚀分析系统

上传海岸照片进行分割分析和空间对齐。系统可以识别包括背景、卵石、沙滩、植物、天空和水等地形类型。""")
        
        # Store analysis content for updating with warnings
        current_analysis = gr.State("")
        outlier_warning = gr.State("")
        
        with gr.Tabs():
            with gr.TabItem("单图像分割"):
                with gr.Row():
                    loc1 = gr.Radio(list(MODEL_PATHS.keys()), label="选择位置", value=list(MODEL_PATHS.keys())[0])
                
                with gr.Row():
                    inp = gr.Image(label="输入图像", type="numpy", image_mode="RGB")
                    seg = gr.Image(label="分割图", type="numpy")
                    ovl = gr.Image(label="叠加可视化", type="numpy")
                
                with gr.Row():
                    btn1 = gr.Button("运行分析", variant="primary")
                
                status1 = gr.HTML(label="异常检测状态")
                res1 = gr.HTML(label="地形分析")
                
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
            
            with gr.TabItem("空间对齐分割"):
                with gr.Row():
                    loc2 = gr.Radio(list(MODEL_PATHS.keys()), label="选择位置", value=list(MODEL_PATHS.keys())[0])
                
                with gr.Row():
                    ref_img = gr.Image(label="参考图像", type="numpy", image_mode="RGB")
                    tgt_img = gr.Image(label="用于分析的目标图像", type="numpy", image_mode="RGB")
                
                with gr.Row():
                    btn2 = gr.Button("运行空间对齐分析", variant="primary")
                
                with gr.Row():
                    orig = gr.Image(label="原始参考", type="numpy")
                    aligned = gr.Image(label="对齐后的图像", type="numpy")
                
                with gr.Row():
                    seg2 = gr.Image(label="分割图", type="numpy")
                    ovl2 = gr.Image(label="叠加可视化", type="numpy")
                
                status2 = gr.HTML(label="空间对齐状态")
                res2 = gr.HTML(label="地形分析")
                
                # For alignment, we use the progressive display function
                btn2.click(
                    fn=run_alignment_and_segmentation,
                    inputs=[loc2, ref_img, tgt_img],
                    outputs=[orig, aligned, seg2, ovl2, res2, status2]
                )
            
            with gr.TabItem("性能监控信息"):
                gr.Markdown("""## 系统性能监控
此选项卡显示有关系统硬件和性能的详细信息。""")
                
                with gr.Row():
                    device_info = gr.HTML(get_system_info())
                
    return demo

def get_system_info():
    """获取系统硬件信息"""
    # CPU信息
    cpu_info = f"CPU核心数: {psutil.cpu_count(logical=True)} ({psutil.cpu_count(logical=False)} 物理核心)"
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        cpu_info += f"<br>CPU频率: {cpu_freq.current:.2f} MHz"
    
    # RAM信息
    ram = psutil.virtual_memory()
    ram_info = f"系统内存: {ram.total/(1024**3):.2f} GB"
    
    # GPU信息
    gpu_info = "GPU: "
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info += f"{gpu_count} CUDA设备可用<br>"
        for i in range(gpu_count):
            gpu_info += f"- {torch.cuda.get_device_name(i)}<br>"
        gpu_info += f"CUDA版本: {torch.version.cuda}"
    else:
        gpu_info += "无可用CUDA设备"
    
    # PyTorch信息
    torch_info = f"PyTorch版本: {torch.__version__}"
    
    # 格式化HTML输出
    html = f"""
    <div style='background-color:#f5f5f5;padding:15px;border-radius:10px;margin:10px 0;'>
        <h3>系统硬件信息</h3>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>
            <div>
                <h4>处理器</h4>
                <p>{cpu_info}</p>
                <h4>内存</h4>
                <p>{ram_info}</p>
            </div>
            <div>
                <h4>图形处理</h4>
                <p>{gpu_info}</p>
                <h4>软件版本</h4>
                <p>{torch_info}</p>
                <p>OpenCV版本: {cv2.__version__}</p>
            </div>
        </div>
    </div>
    """
    return html

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