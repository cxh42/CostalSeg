import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import nn

def load_model(checkpoint_path):
    """从检查点加载模型"""
    class BeachSegmentationModel(LightningModule):
        def __init__(self, arch="DeepLabV3Plus", encoder_name="efficientnet-b4", in_channels=3, out_classes=7, learning_rate=1e-4):
            super().__init__()
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=out_classes,
                encoder_weights="imagenet"
            )
            # 使用组合损失函数：Dice Loss + Focal Loss
            self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
            self.focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2.0)
            self.number_of_classes = out_classes
            self.learning_rate = learning_rate
            self.save_hyperparameters()

        def forward(self, image):
            return self.model(image)

        def shared_step(self, batch, stage):
            image, mask = batch[:2]
            assert image.ndim == 4
            assert mask.ndim == 3
            
            logits_mask = self.forward(image)
            
            dice_loss = self.dice_loss(logits_mask, mask)
            focal_loss = self.focal_loss(logits_mask, mask)
            loss = dice_loss * 0.7 + focal_loss * 0.3
            
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask,
                mask,
                mode="multiclass",
                num_classes=self.number_of_classes,
            )
            
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            
            metrics = {
                "loss": loss,
                "dice_loss": dice_loss,
                "focal_loss": focal_loss,
                "iou_score": iou_score,
                "f1_score": f1_score,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            }
            
            return metrics

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-4
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,
                T_mult=1,
                eta_min=1e-6
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "valid_loss"
                },
            }
    
    # 加载checkpoint
    model = BeachSegmentationModel.load_from_checkpoint(checkpoint_path)
    model.eval()  # 设为评估模式
    
    return model

def segment_images(images, model_path, image_size=1024):    
    # 定义数据增强（调整大小和标准化）
    aug = A.Compose([
        A.Resize(image_size, image_size, always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        )
    ])
    
    # 定义类别与对应的颜色映射（RGB格式）
    COLOR_MAP = {
        0: [0, 0, 0],        # 背景 - 黑色
        1: [194, 178, 128],  # 湿沙 - 沙褐色
        2: [139, 137, 137],  # 卵石 - 深灰色
        3: [255, 228, 181],  # 干沙 - 浅黄色
        4: [135, 206, 235],  # 天空 - 天蓝色
        5: [0, 128, 0],      # 植被 - 绿色
        6: [0, 0, 255]       # 水 - 蓝色
    }
    
    all_segs = []

    model = load_model(model_path)
    
    # 将模型移至可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for i, image in enumerate(images):        
        # 转换为RGB（OpenCV默认为BGR）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 保存Original Image尺寸用于后期处理
        original_height, original_width = image.shape[:2]
        
        # 应用数据增强（调整大小和标准化）
        augmented = aug(image=image)
        image_aug = augmented["image"]
        
        # 转换为模型输入格式（添加批次维度并移至设备）
        x = torch.from_numpy(image_aug.transpose(2, 0, 1)).unsqueeze(0)
        x = x.to(device, dtype=torch.float)
        
        # 模型预测
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_mask = probs.argmax(dim=1).squeeze().cpu().numpy()
        
        # 创建彩色掩码
        color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for cls, color in COLOR_MAP.items():
            color_mask[pred_mask == cls] = color

        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        all_segs.append(color_mask)
        
    return all_segs

