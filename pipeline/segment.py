import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import nn

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
        self.ids = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # Modify mask file path construction
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('.jpg', '_mask.png')) for image_id in self.ids]
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

def get_validation_augmentation(image_size=1024):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class BeachSegmentationModel(pl.LightningModule):
    def __init__(self, arch="DeepLabV3Plus", encoder_name="efficientnet-b7", in_channels=3, out_classes=7, learning_rate=1e-4):
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

        return optimizer


def segment_images(images, model_path, image_size=1024, device=None):
    """
    Segment a list of images using a pre-trained beach segmentation model.

    Args:
        images: List of images opened with cv2 (BGR format)
        model_path: Path to the saved model checkpoint
        image_size: Size to resize images to before segmentation
        device: Device to run inference on ('cuda', 'cpu', etc.)

    Returns:
        List of segmented images with color-coded classes
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = BeachSegmentationModel.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()

    # Define the preprocessing transform
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Process each image
    segmented_images = []

    with torch.no_grad():
        for img in images:
            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply preprocessing
            transformed = transform(image=img_rgb)
            img_tensor = torch.from_numpy(transformed['image'].transpose(2, 0, 1)).float().unsqueeze(0)
            img_tensor = img_tensor.to(device)

            # Get model prediction
            logits = model(img_tensor)
            prob_mask = logits.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1).cpu().numpy().squeeze()

            # Create color-coded segmentation mask
            segmentation_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

            for class_idx, color in BeachSegmentationDataset.COLOR_MAP.items():
                segmentation_mask[pred_mask == class_idx] = color

            # Resize back to original image size if needed
            if img.shape[:2] != segmentation_mask.shape[:2]:
                segmentation_mask = cv2.resize(
                    segmentation_mask, 
                    (img.shape[1], img.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )

            segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)


            segmented_images.append(segmentation_mask)

    return segmented_images


def overlay_segmentation(original_images, segmentation_masks, alpha=0.5):
    """
    Overlay segmentation masks on original images.

    Args:
        original_images: List of original images (BGR format)
        segmentation_masks: List of segmentation masks (RGB format)
        alpha: Transparency factor for overlay

    Returns:
        List of images with overlaid segmentation
    """
    overlaid_images = []

    for img, mask in zip(original_images, segmentation_masks):
        # Blend images
        overlaid = cv2.addWeighted(img, 1-alpha, mask_bgr, alpha, 0)
        overlaid_images.append(overlaid)

    return overlaid_images

