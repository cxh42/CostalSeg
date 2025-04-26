import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class BeachSegmentationDataset(Dataset):
    """Dataset class for beach scene segmentation"""
    
    CLASSES = [
        "background",
        "cobbles",
        "drysand",
        "plant",
        "sky",
        "water",
        "wetsand"
    ]
    
    # Define color mapping for different classes, used for visualization
    COLOR_MAP = {
        0: [0, 0, 0],        # background - black
        1: [139, 137, 137],  # cobbles - dark gray
        2: [255, 228, 181],  # dry sand - light yellow
        3: [0, 128, 0],      # plant - green
        4: [135, 206, 235],  # sky - sky blue
        5: [0, 0, 255],      # water - blue
        6: [194, 178, 128]   # wet sand - sandy brown
    }

    def __init__(self, images_dir, masks_dir, augmentation=None):
        # 检查目录是否存在
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"掩码目录不存在: {masks_dir}")
            
        # 获取所有jpg文件
        self.ids = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        # 如果没有找到jpg文件，尝试查找其他图像格式
        if len(self.ids) == 0:
            print(f"警告: 在 {images_dir} 中没有找到jpg文件")
            print("尝试查找png文件...")
            self.ids = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            if len(self.ids) > 0:
                print(f"找到 {len(self.ids)} 个png文件")
        
        # 如果还是没有找到图像文件，抛出错误
        if len(self.ids) == 0:
            raise FileNotFoundError(f"在 {images_dir} 中没有找到支持的图像文件 (jpg/png)")
        else:
            print(f"找到 {len(self.ids)} 个图像文件")
            
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        # 构建掩码文件路径 (支持jpg和png)
        if self.ids[0].endswith('.jpg'):
            self.masks_fps = [os.path.join(masks_dir, image_id.replace('.jpg', '_mask.png')) for image_id in self.ids]
        else:
            self.masks_fps = [os.path.join(masks_dir, image_id.replace('.png', '_mask.png')) for image_id in self.ids]
            
        # 检查第一个掩码文件是否存在
        if not os.path.exists(self.masks_fps[0]):
            print(f"警告: 第一个掩码文件不存在: {self.masks_fps[0]}")
            print("尝试不同的掩码文件命名格式...")
            
            # 尝试其他常见的掩码命名格式
            if self.ids[0].endswith('.jpg'):
                test_mask = os.path.join(masks_dir, self.ids[0].replace('.jpg', '.png'))
            else:
                test_mask = os.path.join(masks_dir, self.ids[0].replace('.png', '.png'))
                
            if os.path.exists(test_mask):
                print(f"找到替代掩码格式: {test_mask}")
                if self.ids[0].endswith('.jpg'):
                    self.masks_fps = [os.path.join(masks_dir, image_id.replace('.jpg', '.png')) for image_id in self.ids]
                else:
                    self.masks_fps = [os.path.join(masks_dir, image_id.replace('.png', '.png')) for image_id in self.ids]
        
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # No need to merge classes, use original annotations
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        # Transpose to CHW format and convert to float32 type
        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.astype('int64')
        return image, mask, self.images_fps[i]  # Add return image file path for visualization

    def __len__(self):
        return len(self.ids)

def get_training_augmentation(image_size=1024):
    # Remove always_apply parameter
    train_transform = [
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(image_size=1024):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class BeachSegmentationModel(pl.LightningModule):
    def __init__(self, arch="DeepLabV3Plus", encoder_name="efficientnet-b6", in_channels=3, out_classes=7, learning_rate=1e-4):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights="imagenet"  # Use pretrained weights
        )
        # Use combined loss function: Dice Loss + Focal Loss to improve segmentation accuracy
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2.0)
        self.number_of_classes = out_classes
        self.learning_rate = learning_rate
        
        # Save hyperparameters for later loading
        self.save_hyperparameters()

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch[:2]  # Only take the first two elements, ignore file path
        assert image.ndim == 4
        assert mask.ndim == 3
        
        # Get model predictions
        logits_mask = self.forward(image)
        
        # Calculate composite loss
        dice_loss = self.dice_loss(logits_mask, mask)
        focal_loss = self.focal_loss(logits_mask, mask)
        loss = dice_loss * 0.7 + focal_loss * 0.3  # Weighted combination
        
        # Calculate performance metrics
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask,
            mask,
            mode="multiclass",
            num_classes=self.number_of_classes,
        )
        
        # Calculate IoU and F1 scores
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        
        # Record all metrics
        metrics = {
            "loss": loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "iou_score": iou_score,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, "train")
        # Log training metrics
        self.log("train_loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", metrics["iou_score"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", metrics["f1_score"], on_step=False, on_epoch=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, "valid")
        # Log validation metrics
        self.log("valid_loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_iou", metrics["iou_score"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_f1", metrics["f1_score"], on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, "test")
        # Log test metrics
        self.log("test_loss", metrics["loss"], on_step=False, on_epoch=True)
        self.log("test_iou", metrics["iou_score"], on_step=False, on_epoch=True)
        self.log("test_f1", metrics["f1_score"], on_step=False, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        # Use AdamW optimizer, usually performs better than Adam for image tasks
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Use cosine annealing warm restarts scheduler
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # First restart cycle length
            T_mult=1,  # Subsequent cycle coefficient
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "valid_loss"
            },
        }

def visualize_predictions(model, dataset, device, num_samples=4, save_dir="predictions"):
    """Visualize model prediction results"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for i, (image, mask, filepath) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Move data to device
            image = image.to(device)
            mask = mask.to(device)
            
            # Get predictions
            logits = model(image)
            prob_mask = logits.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)
            
            # Convert to CPU and numpy
            image = image.cpu().numpy().squeeze().transpose(1, 2, 0)
            mask = mask.cpu().numpy().squeeze()
            pred_mask = pred_mask.cpu().numpy().squeeze()
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            
            # Create color masks
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            color_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            
            for cls, color in dataset.COLOR_MAP.items():
                color_mask[mask == cls] = color
                color_pred_mask[pred_mask == cls] = color
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(color_mask)
            plt.title("Ground Truth")
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(color_pred_mask)
            plt.title("Prediction")
            plt.axis('off')
            
            # Get filename
            filename = os.path.basename(filepath[0])
            plt.savefig(os.path.join(save_dir, f"prediction_{filename}.png"))
            plt.close()

def load_torch_model(model_path, device="cpu", arch="DeepLabV3Plus", encoder_name="efficientnet-b6", in_channels=3, out_classes=7):
    """加载标准PyTorch格式的模型(.pth)用于推理"""
    # 创建模型架构
    model = smp.create_model(
        arch,
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=out_classes,
        encoder_weights=None  # 不加载预训练权重
    )
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def train_model(data_dir="./data", image_size=1024, batch_size=4, epochs=100, num_workers=0):
    # Set data directories
    x_train_dir = os.path.join(data_dir, "train")
    y_train_dir = os.path.join(data_dir, "train")
    x_valid_dir = os.path.join(data_dir, "valid")
    y_valid_dir = os.path.join(data_dir, "valid")
    x_test_dir = os.path.join(data_dir, "test")
    y_test_dir = os.path.join(data_dir, "test")
    
    # 打印数据目录信息，方便调试
    print(f"使用以下数据目录:")
    print(f"训练图像目录: {x_train_dir}")
    print(f"验证图像目录: {x_valid_dir}")
    print(f"测试图像目录: {x_test_dir}")

    # Create datasets
    train_dataset = BeachSegmentationDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(image_size),
    )
    valid_dataset = BeachSegmentationDataset(
        x_valid_dir, y_valid_dir,
        augmentation=get_validation_augmentation(image_size),
    )
    test_dataset = BeachSegmentationDataset(
        x_test_dir, y_test_dir,
        augmentation=get_validation_augmentation(image_size),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    # Create model
    model = BeachSegmentationModel(
        arch="DeepLabV3Plus",
        encoder_name="efficientnet-b6",  # Use EfficientNetB6
        in_channels=3,
        out_classes=len(train_dataset.CLASSES),
        learning_rate=8e-5  # Adjusted learning rate for EfficientNet B6 with small batch size
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_iou",  # Monitor IoU instead of loss
        mode="max",  # Maximize IoU 
        filename="best_model-{epoch:02d}-{valid_iou:.4f}",
        save_top_k=3,
        verbose=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        patience=15,  # Stop if no improvement for 15 epochs 
        mode="min",
        verbose=True
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision training for acceleration
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0,  # Gradient clipping to prevent explosion
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # Test model
    test_results = trainer.test(model, dataloaders=test_loader)
    
    # 保存为标准PyTorch格式(.pth)
    best_model = BeachSegmentationModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch_model_path = os.path.splitext(checkpoint_callback.best_model_path)[0] + ".pth"
    
    # 仅保存模型的权重部分（state_dict）
    torch.save(best_model.model.state_dict(), torch_model_path)
    print(f"标准PyTorch模型保存为: {torch_model_path}")
    
    # 也可以保存完整模型（通常文件更大）
    full_model_path = os.path.splitext(checkpoint_callback.best_model_path)[0] + "_full.pth"
    torch.save(best_model.model, full_model_path)
    print(f"完整PyTorch模型保存为: {full_model_path}")
    
    # Visualize some prediction results
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visualize_predictions(model, test_dataset, device, num_samples=8, save_dir="prediction_results")
    
    return model, trainer, checkpoint_callback.best_model_path, test_results, torch_model_path, full_model_path

if __name__ == "__main__":
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # 创建保存模型的目录
    os.makedirs("models", exist_ok=True)
    
    # 确保数据目录路径正确 - 使用相对路径，train.py同级目录下的data文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "dataset")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录 {data_dir} 不存在!")
        print("请确保dataset目录包含train, valid和test子文件夹")
    else:
        print(f"找到数据目录: {data_dir}")
        # 检查子目录
        for subdir in ["train", "valid", "test"]:
            subdir_path = os.path.join(data_dir, subdir)
            if not os.path.exists(subdir_path):
                print(f"警告: 子目录 {subdir_path} 不存在!")
            else:
                print(f"找到子目录: {subdir_path}")
    
    # Start training
    model, trainer, best_model_path, test_results, torch_model_path, full_model_path = train_model(
        data_dir=data_dir,
        image_size=1024,
        batch_size=2,  # Reduced batch size to 2
        epochs=100,
        num_workers=0  # Use 0 for Windows systems
    )
    
    print(f"Best model saved at: {best_model_path}")
    print(f"标准PyTorch模型(.pth)保存为: {torch_model_path}")
    print(f"完整PyTorch模型保存为: {full_model_path}")
    print(f"Test results: IoU={test_results[0]['test_iou']:.4f}, F1={test_results[0]['test_f1']:.4f}")
    print("Training completed!")
    
    # 演示如何加载和使用标准PyTorch模型进行推理
    print("\n演示如何加载和使用标准PyTorch模型:")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载保存的模型
    loaded_model = load_torch_model(
        torch_model_path, 
        device=device,
        arch="DeepLabV3Plus",
        encoder_name="efficientnet-b6",
        in_channels=3,
        out_classes=7
    )
    
    print("模型加载成功，可以用于推理。")