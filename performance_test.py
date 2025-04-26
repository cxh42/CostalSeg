import os
import time
import gc
import torch
import numpy as np
import cv2
import psutil
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchinfo import summary
import platform
import json
from datetime import datetime

# Try to import GPU monitoring libraries
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Try to import FLOP calculation library
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

class ModelTester:
    """Universal model performance testing class"""
    
    def __init__(self, model_path, output_dir="performance_results"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
        self.system_info = self._get_system_info()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Running on device: {self.device}")
        print(f"System information: {self.system_info}")

    def _get_system_info(self):
        """Get detailed system information"""
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cpu_count": os.cpu_count(),
            "ram_total": round(psutil.virtual_memory().total / (1024**3), 2)  # GB
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_count"] = torch.cuda.device_count()
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            
            # Try to get more detailed GPU info using NVML
            if NVML_AVAILABLE:
                try:
                    nvmlInit()
                    handle = nvmlDeviceGetHandleByIndex(0)
                    system_info["gpu_name_nvml"] = nvmlDeviceGetName(handle)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    system_info["gpu_memory_total"] = round(mem_info.total / (1024**3), 2)  # GB
                except Exception as e:
                    print(f"Error getting detailed GPU info: {e}")
        
        return system_info

    def get_process_memory(self):
        """Get current process memory usage (RAM)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss  # Return bytes

    def get_gpu_memory(self, device_id=0):
        """Get GPU memory usage"""
        if not NVML_AVAILABLE or not torch.cuda.is_available():
            return 0
        
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device_id)
            info = nvmlDeviceGetMemoryInfo(handle)
            return info.used  # Return bytes
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            return 0

    def load_model(self):
        """Load the model with proper error handling"""
        try:
            # Add safe globals to allow DeepLabV3Plus class loading
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
            import torch.serialization
            
            # Allow DeepLabV3Plus class in PyTorch 2.6+
            try:
                torch.serialization.add_safe_globals([DeepLabV3Plus])
                print("Added DeepLabV3Plus to safe globals")
            except (AttributeError, ImportError):
                print("Current PyTorch version doesn't support add_safe_globals, will try weights_only=False")
            
            # Try safe loading first
            try:
                model = torch.load(self.model_path, map_location=self.device)
                print(f"Successfully loaded complete model: {self.model_path}")
            except Exception as e1:
                print(f"Safe loading failed, trying with weights_only=False: {e1}")
                # If safe loading fails, use non-safe method
                model = torch.load(self.model_path, map_location=self.device, weights_only=False)
                print(f"Successfully loaded model with weights_only=False")
        except Exception as e:
            print(f"Loading complete model failed: {e}")
            print("Trying to create model and load weights...")
            
            # Create model architecture
            model = smp.create_model(
                "DeepLabV3Plus",
                encoder_name="efficientnet-b6",
                in_channels=3,
                classes=7,
                encoder_weights=None
            )
            
            # Try different loading methods
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                print("Loaded weights file")
            except Exception as e2:
                print(f"Trying to load weights with weights_only=False")
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # If loaded a complete model with state_dict
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
                # If keys start with 'model.', remove the prefix
                if all(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {k[6:]: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            print(f"Successfully created model and loaded weights")
        
        model.to(self.device)
        model.eval()
        return model

    def generate_test_image(self, size=(1024, 1024)):
        """Generate random test image"""
        img = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
        return img

    def prepare_input(self, image):
        """Convert image to model input format"""
        # Convert to RGB if needed
        if image.shape[2] == 4:  # If RGBA, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        if image.shape[0] != 1024 or image.shape[1] != 1024:
            image = cv2.resize(image, (1024, 1024))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to PyTorch tensor and add batch dimension
        image = image.transpose(2, 0, 1).astype('float32')
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        return tensor

    def test_inference_performance(self, model, image_size=(1024, 1024), device=None, warmup=5, iterations=10):
        """Test model inference performance on a single image"""
        if device is None:
            device = self.device
            
        print(f"\nTesting inference performance on {device} with image size {image_size}:")
        
        # Generate test image
        image = self.generate_test_image(image_size)
        
        # Warm up the model
        print("Warming up model...")
        input_tensor = self.prepare_input(image).to(device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Clear any cache
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Record initial memory usage
        start_cpu_mem = self.get_process_memory()
        start_gpu_mem = self.get_gpu_memory() if device == "cuda" else 0
        
        # Measure inference time
        inference_times = []
        print(f"Running {iterations} inference iterations...")
        
        for i in range(iterations):
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                output = model(input_tensor)
                
            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{iterations} iterations")
        
        # Measure memory usage
        end_cpu_mem = self.get_process_memory()
        end_gpu_mem = self.get_gpu_memory() if device == "cuda" else 0
        
        # Calculate results
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_dev = np.std(inference_times)
        
        # Calculate 95% confidence interval
        inference_times.sort()
        idx_95 = int(0.95 * len(inference_times))
        percentile_95 = inference_times[idx_95] if idx_95 < len(inference_times) else max_time
        
        cpu_mem_used = (end_cpu_mem - start_cpu_mem) / (1024 * 1024)  # MB
        gpu_mem_used = (end_gpu_mem - start_gpu_mem) / (1024 * 1024) if device == "cuda" else 0  # MB
        
        # Print results
        print(f"\nResults:")
        print(f"Average inference time: {avg_time:.4f} seconds ({1/avg_time:.2f} FPS)")
        print(f"Minimum inference time: {min_time:.4f} seconds ({1/min_time:.2f} FPS)")
        print(f"Maximum inference time: {max_time:.4f} seconds ({1/max_time:.2f} FPS)")
        print(f"Standard deviation: {std_dev:.4f} seconds")
        print(f"95th percentile inference time: {percentile_95:.4f} seconds ({1/percentile_95:.2f} FPS)")
        print(f"CPU memory usage: {cpu_mem_used:.2f} MB")
        
        if device == "cuda":
            print(f"GPU memory usage: {gpu_mem_used:.2f} MB")
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "percentile_95": percentile_95,
            "cpu_mem_used": cpu_mem_used,
            "gpu_mem_used": gpu_mem_used if device == "cuda" else 0,
            "fps": 1/avg_time,
            "min_fps": 1/max_time,
            "max_fps": 1/min_time
        }

    def test_batch_performance(self, model, device=None, batch_sizes=[1, 2, 4, 8], iterations=5):
        """Test performance with different batch sizes"""
        if device is None:
            device = self.device
            
        if device != "cuda":
            print("Warning: Batch performance testing is best performed on GPU")
            batch_sizes = [1, 2]  # Limit batch sizes for CPU testing
        
        print(f"\nTesting batch performance (device={device}):")
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Create batch input
            images = [self.generate_test_image() for _ in range(batch_size)]
            tensors = [self.prepare_input(img).to(device) for img in images]
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Warm up
            print("Warming up...")
            with torch.no_grad():
                _ = model(batch_tensor)
            
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Record initial memory
            start_gpu_mem = self.get_gpu_memory() if device == "cuda" else 0
            
            # Measure time
            batch_times = []
            print(f"Running {iterations} inference passes...")
            
            for i in range(iterations):
                if device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(batch_tensor)
                    
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                batch_times.append(end_time - start_time)
                print(f"Batch {i+1}/{iterations}: {end_time - start_time:.4f} seconds")
            
            # Record final memory
            end_gpu_mem = self.get_gpu_memory() if device == "cuda" else 0
            
            # Calculate results
            avg_time = sum(batch_times) / len(batch_times)
            min_time = min(batch_times)
            images_per_second = batch_size / avg_time
            gpu_mem_used = (end_gpu_mem - start_gpu_mem) / (1024 * 1024) if device == "cuda" else 0
            
            print(f"Average time for batch size {batch_size}: {avg_time:.4f} seconds")
            print(f"Images processed per second: {images_per_second:.2f}")
            if device == "cuda":
                print(f"GPU memory usage: {gpu_mem_used:.2f} MB")
            
            # Calculate efficiency metrics
            efficiency_per_instance = avg_time / batch_size  # Time per image in batch
            scaling_efficiency = (batch_times[0] * batch_size) / (batch_times[-1] * 1)  # Perfect scaling would be 1.0
            
            results[batch_size] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "time_per_image": efficiency_per_instance,
                "images_per_second": images_per_second,
                "scaling_efficiency": scaling_efficiency,
                "gpu_mem_used": gpu_mem_used,
                "mem_per_instance": gpu_mem_used / batch_size if gpu_mem_used > 0 else 0
            }
        
        return results

    def get_model_info(self, model, input_size=(1, 3, 1024, 1024)):
        """Get detailed model information"""
        print("\nGetting model information:")
        model_stats = summary(model, input_size=input_size, verbose=0)
        print(model_stats)
        
        # Extract key information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nTotal model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_summary": str(model_stats)
        }

    def calculate_flops(self, model, input_size=(1, 3, 1024, 1024), device=None):
        """Calculate model FLOPs (floating point operations)"""
        if device is None:
            device = self.device
            
        if not THOP_AVAILABLE:
            print("\nCalculating FLOPs requires the thop library. Install with pip install thop")
            return {"flops": 0, "params": 0}
        
        print("\nCalculating model FLOPs:")
        input_tensor = torch.randn(input_size).to(device)
        model.to(device)
        
        flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
        
        print(f"Model FLOPs: {flops/1e9:.2f} G")
        print(f"Model parameters: {params/1e6:.2f} M")
        
        # Calculate normalized metrics
        flops_per_param = flops / params if params > 0 else 0
        
        return {
            "flops": flops,
            "params": params,
            "flops_per_param": flops_per_param,
            "flops_G": flops/1e9,
            "params_M": params/1e6
        }

    def test_size_performance(self, model, device=None, sizes=[(512, 512), (1024, 1024), (2048, 2048)]):
        """Test performance with different input sizes"""
        if device is None:
            device = self.device
            
        print(f"\nTesting performance with different image sizes (device={device}):")
        results = {}
        
        # Use smaller iteration counts to speed up testing
        warmup = 2
        iterations = 5
        
        for size in sizes:
            try:
                print(f"\nImage size: {size}")
                result = self.test_inference_performance(
                    model, image_size=size, device=device, 
                    warmup=warmup, iterations=iterations
                )
                
                # Calculate additional metrics
                pixels = size[0] * size[1]
                time_per_mpixel = result["avg_time"] / (pixels / 1e6)
                mem_per_mpixel = result["gpu_mem_used"] / (pixels / 1e6) if result["gpu_mem_used"] > 0 else 0
                
                result["pixels"] = pixels
                result["time_per_mpixel"] = time_per_mpixel
                result["mem_per_mpixel"] = mem_per_mpixel
                
                results[f"{size[0]}x{size[1]}"] = result
            except Exception as e:
                print(f"Error testing size {size}: {e}")
                results[f"{size[0]}x{size[1]}"] = {"error": str(e)}
        
        return results

    def calculate_universal_metrics(self, single_image_results, batch_results, flops_results, model_info):
        """Calculate hardware-agnostic universal metrics"""
        universal_metrics = {}
        
        # Basic metrics
        if 'cuda' in single_image_results:
            gpu_fps = single_image_results['cuda']['fps']
            gpu_mem = single_image_results['cuda']['gpu_mem_used']
        else:
            gpu_fps = 0
            gpu_mem = 0
            
        total_params = model_info['total_params']
        flops = flops_results.get('flops', 0) if flops_results else 0
        
        # Efficiency metrics
        if gpu_fps > 0 and gpu_mem > 0:
            fps_per_mb = gpu_fps / gpu_mem
            universal_metrics['fps_per_mb'] = fps_per_mb
        
        if flops > 0:
            fps_per_gflops = gpu_fps / (flops/1e9) if gpu_fps > 0 else 0
            universal_metrics['fps_per_gflops'] = fps_per_gflops
        
        if total_params > 0:
            fps_per_mparam = gpu_fps / (total_params/1e6) if gpu_fps > 0 else 0
            universal_metrics['fps_per_mparam'] = fps_per_mparam
        
        # Batch processing efficiency
        if batch_results and 8 in batch_results:
            batch_scaling = batch_results[1]['images_per_second'] / (batch_results[8]['images_per_second'] / 8)
            universal_metrics['batch_scaling_efficiency'] = batch_scaling
        
        # Memory scaling analysis
        if batch_results and 1 in batch_results and 8 in batch_results:
            mem_overhead = (batch_results[8]['gpu_mem_used'] / 8) / batch_results[1]['gpu_mem_used']
            universal_metrics['memory_scaling_overhead'] = mem_overhead
        
        return universal_metrics

    def run_all_tests(self):
        """Run all performance tests and generate reports"""
        print(f"Starting comprehensive performance testing for model: {self.model_path}")
        
        # Load the model
        try:
            model = self.load_model()
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
        
        # Get model information
        self.results['model_info'] = self.get_model_info(model)
        
        # Calculate FLOPs
        if THOP_AVAILABLE:
            self.results['flops'] = self.calculate_flops(model, device=self.device)
        
        # Test single image performance
        self.results['single_image'] = {}
        
        # Test on CPU
        print("\nTesting on CPU...")
        model.to("cpu")
        self.results['single_image']['cpu'] = self.test_inference_performance(
            model, device="cpu", warmup=2, iterations=10
        )
        
        # Test on GPU if available
        if torch.cuda.is_available():
            print("\nTesting on GPU...")
            model.to("cuda")
            self.results['single_image']['cuda'] = self.test_inference_performance(
                model, device="cuda", warmup=5, iterations=20
            )
            
            # Test batch performance (GPU only)
            self.results['batch_performance'] = self.test_batch_performance(
                model, device="cuda", batch_sizes=[1, 2, 4, 8]
            )
            
            # Test different image sizes
            self.results['size_performance'] = self.test_size_performance(
                model, device="cuda", sizes=[(512, 512), (1024, 1024), (1536, 1536)]
            )
        else:
            print("No GPU available, skipping GPU tests")
            
            # Simplified batch testing on CPU
            self.results['batch_performance'] = self.test_batch_performance(
                model, device="cpu", batch_sizes=[1, 2], iterations=3
            )
            
            # Test different image sizes on CPU (use smaller sizes)
            self.results['size_performance'] = self.test_size_performance(
                model, device="cpu", sizes=[(320, 320), (512, 512), (1024, 1024)]
            )
        
        # Calculate universal metrics
        self.results['universal_metrics'] = self.calculate_universal_metrics(
            self.results['single_image'],
            self.results['batch_performance'],
            self.results.get('flops', None),
            self.results['model_info']
        )
        
        # Add system information
        self.results['system_info'] = self.system_info
        self.results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save results
        self.save_results()
        
        # Generate charts
        self.generate_performance_charts()
        
        print("\nPerformance testing completed!")
        
    def save_results(self, filename=None):
        """Save test results to files"""
        if filename is None:
            filename = os.path.join(self.output_dir, "model_performance_results.txt")
        
        # Save as text file
        with open(filename, "w") as f:
            f.write("Beach Segmentation Model Performance Test Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Write system information
            f.write("System Information:\n")
            for key, value in self.results['system_info'].items():
                f.write(f"{key}: {value}\n")
            
            # Write model information
            f.write("\nModel Information:\n")
            f.write(f"Total parameters: {self.results['model_info']['total_params']:,}\n")
            f.write(f"Trainable parameters: {self.results['model_info']['trainable_params']:,}\n")
            
            # Write FLOPs information
            if 'flops' in self.results:
                f.write(f"\nFLOPs: {self.results['flops']['flops_G']:.2f} G\n")
            
            # Write single image inference results
            for device in ['cpu', 'cuda']:
                if device in self.results['single_image']:
                    f.write(f"\n{device.upper()} Single Image Inference Performance:\n")
                    perf = self.results['single_image'][device]
                    f.write(f"Average inference time: {perf['avg_time']:.4f} seconds ({perf['fps']:.2f} FPS)\n")
                    f.write(f"Minimum inference time: {perf['min_time']:.4f} seconds ({perf['max_fps']:.2f} FPS)\n")
                    f.write(f"Maximum inference time: {perf['max_time']:.4f} seconds\n")
                    f.write(f"Standard deviation: {perf['std_dev']:.4f} seconds\n")
                    f.write(f"95th percentile time: {perf['percentile_95']:.4f} seconds\n")
                    f.write(f"CPU memory usage: {perf['cpu_mem_used']:.2f} MB\n")
                    if device == 'cuda':
                        f.write(f"GPU memory usage: {perf['gpu_mem_used']:.2f} MB\n")
            
            # Write batch processing results
            if 'batch_performance' in self.results:
                f.write("\nBatch Processing Performance:\n")
                for batch_size, perf in self.results['batch_performance'].items():
                    f.write(f"Batch size {batch_size}:\n")
                    f.write(f"  Average time: {perf['avg_time']:.4f} seconds\n")
                    f.write(f"  Time per image: {perf['time_per_image']:.4f} seconds\n")
                    f.write(f"  Images per second: {perf['images_per_second']:.2f}\n")
                    f.write(f"  Scaling efficiency: {perf['scaling_efficiency']:.2f}\n")
                    if 'gpu_mem_used' in perf and perf['gpu_mem_used'] > 0:
                        f.write(f"  GPU memory usage: {perf['gpu_mem_used']:.2f} MB\n")
                        f.write(f"  Memory per image: {perf['mem_per_instance']:.2f} MB\n")
            
            # Write different image size performance results
            if 'size_performance' in self.results:
                f.write("\nPerformance with Different Image Sizes:\n")
                for size, perf in self.results['size_performance'].items():
                    if 'error' in perf:
                        f.write(f"Size {size}: Error - {perf['error']}\n")
                    else:
                        f.write(f"Size {size}:\n")
                        f.write(f"  Average inference time: {perf['avg_time']:.4f} seconds ({perf['fps']:.2f} FPS)\n")
                        f.write(f"  Time per megapixel: {perf['time_per_mpixel']:.4f} seconds\n")
                        if 'gpu_mem_used' in perf and perf['gpu_mem_used'] > 0:
                            f.write(f"  GPU memory usage: {perf['gpu_mem_used']:.2f} MB\n")
                            f.write(f"  Memory per megapixel: {perf['mem_per_mpixel']:.2f} MB\n")
            
            # Write universal metrics
            if 'universal_metrics' in self.results:
                f.write("\nUniversal Performance Metrics:\n")
                metrics = self.results['universal_metrics']
                if 'fps_per_mb' in metrics:
                    f.write(f"FPS per MB of GPU memory: {metrics['fps_per_mb']:.4f}\n")
                if 'fps_per_gflops' in metrics:
                    f.write(f"FPS per GFLOPS: {metrics['fps_per_gflops']:.4f}\n")
                if 'fps_per_mparam' in metrics:
                    f.write(f"FPS per million parameters: {metrics['fps_per_mparam']:.4f}\n")
                if 'batch_scaling_efficiency' in metrics:
                    f.write(f"Batch scaling efficiency (1→8): {metrics['batch_scaling_efficiency']:.2f}\n")
                if 'memory_scaling_overhead' in metrics:
                    f.write(f"Memory scaling overhead: {metrics['memory_scaling_overhead']:.2f}\n")
        
        # Save full results as JSON for programmatic analysis
        json_filename = os.path.join(self.output_dir, "model_performance_results.json")
        with open(json_filename, "w") as f:
            # Convert to serializable format
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            serializable_results[key][k] = {}
                            for kk, vv in v.items():
                                if hasattr(vv, 'tolist'):  # Handle numpy arrays
                                    serializable_results[key][k][kk] = vv.tolist()
                                elif isinstance(vv, torch.Tensor):  # Handle torch tensors
                                    serializable_results[key][k][kk] = vv.cpu().numpy().tolist()
                                else:
                                    serializable_results[key][k][kk] = vv
                        elif hasattr(v, 'tolist'):  # Handle numpy arrays
                            serializable_results[key][k] = v.tolist()
                        elif isinstance(v, torch.Tensor):  # Handle torch tensors
                            serializable_results[key][k] = v.cpu().numpy().tolist()
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filename} and {json_filename}")

    def generate_performance_charts(self, output_dir=None):
        """Generate performance comparison charts"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "charts")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. CPU vs GPU performance comparison
        if 'cpu' in self.results['single_image'] and 'cuda' in self.results['single_image']:
            plt.figure(figsize=(10, 6))
            devices = ['CPU', 'GPU']
            times = [
                self.results['single_image']['cpu']['avg_time'],
                self.results['single_image']['cuda']['avg_time']
            ]
            fps = [1/t for t in times]
            
            plt.subplot(1, 2, 1)
            bars = plt.bar(devices, times, color=['blue', 'orange'])
            plt.title('Average Inference Time (seconds)')
            plt.ylabel('Time (seconds)')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f"{height:.4f}", ha='center', va='bottom')
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(devices, fps, color=['blue', 'orange'])
            plt.title('Frames Per Second (FPS)')
            plt.ylabel('FPS')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f"{height:.2f}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cpu_vs_gpu.png'))
            plt.close()
        
        # 2. Batch size performance
        if 'batch_performance' in self.results:
            batch_sizes = list(self.results['batch_performance'].keys())
            avg_times = [self.results['batch_performance'][bs]['avg_time'] for bs in batch_sizes]
            imgs_per_sec = [self.results['batch_performance'][bs]['images_per_second'] for bs in batch_sizes]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            bars = plt.bar(batch_sizes, avg_times)
            plt.title('Average Batch Processing Time')
            plt.xlabel('Batch Size')
            plt.ylabel('Time (seconds)')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f"{height:.4f}", ha='center', va='bottom')
            
            plt.subplot(1, 3, 2)
            bars = plt.bar(batch_sizes, imgs_per_sec)
            plt.title('Images Processed Per Second')
            plt.xlabel('Batch Size')
            plt.ylabel('Images/second')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f"{height:.2f}", ha='center', va='bottom')
            
            # Add efficiency plot
            if len(batch_sizes) > 1:
                efficiency = [self.results['batch_performance'][bs]['scaling_efficiency'] for bs in batch_sizes]
                
                plt.subplot(1, 3, 3)
                bars = plt.bar(batch_sizes, efficiency)
                plt.title('Scaling Efficiency')
                plt.xlabel('Batch Size')
                plt.ylabel('Efficiency')
                plt.ylim(0, max(2, max(efficiency)))
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f"{height:.2f}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'batch_performance.png'))
            plt.close()
        
        # 3. Image size performance
        if 'size_performance' in self.results:
            sizes = []
            avg_times = []
            fps_values = []
            time_per_mpixel = []
            
            for size, perf in self.results['size_performance'].items():
                if 'error' not in perf:
                    sizes.append(size)
                    avg_times.append(perf['avg_time'])
                    fps_values.append(perf['fps'])
                    time_per_mpixel.append(perf['time_per_mpixel'])
            
            if sizes:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                bars = plt.bar(sizes, avg_times)
                plt.title('Average Inference Time by Size')
                plt.xlabel('Image Size')
                plt.ylabel('Time (seconds)')
                plt.xticks(rotation=45)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f"{height:.4f}", ha='center', va='bottom')
                
                plt.subplot(1, 3, 2)
                bars = plt.bar(sizes, fps_values)
                plt.title('FPS by Image Size')
                plt.xlabel('Image Size')
                plt.ylabel('FPS')
                plt.xticks(rotation=45)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f"{height:.2f}", ha='center', va='bottom')
                
                plt.subplot(1, 3, 3)
                bars = plt.bar(sizes, time_per_mpixel)
                plt.title('Processing Time per Megapixel')
                plt.xlabel('Image Size')
                plt.ylabel('Time per Mpixel (seconds)')
                plt.xticks(rotation=45)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f"{height:.4f}", ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'size_performance.png'))
                plt.close()
        
        # 4. Universal metrics visualization
        if 'universal_metrics' in self.results:
            metrics = self.results['universal_metrics']
            metric_names = []
            metric_values = []
            
            for key, value in metrics.items():
                if key in ['fps_per_mb', 'fps_per_gflops', 'fps_per_mparam']:
                    metric_names.append(key)
                    metric_values.append(value)
            
            if metric_names:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(metric_names, metric_values)
                plt.title('Universal Efficiency Metrics')
                plt.ylabel('Efficiency Value')
                plt.xticks(rotation=45)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f"{height:.4f}", ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'universal_metrics.png'))
                plt.close()
        
        print(f"Performance charts saved to {output_dir}")


if __name__ == "__main__":
    # Set model path
    model_path = r"D:\Programs\CostalSeg\lightning_logs\version_0\checkpoints\best_model-epoch=24-valid_iou=0.9270_full.pth"
    
    # Create tester instance
    tester = ModelTester(model_path, output_dir="performance_results")
    
    # Run all tests
    tester.run_all_tests()