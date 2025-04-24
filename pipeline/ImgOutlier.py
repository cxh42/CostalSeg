import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import vit_h_14
import cv2

class CosineSimilarity:
    def __init__(self, vector='feature', threshold=0.8, mean_vec=[], device=None):
        """
        Initialize the CosineSimilarity class.

        Args:
            vector (str): Type of vector to use ('feature' or 'image')
            threshold (float): Threshold for determining outliers
            mean_vec (numpy vector): Preloaded reference vector for comparison
            device (str): Device to use for computation (default: 'mps' if available, else 'cuda' if available, else 'cpu')
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.vector = vector
        self.threshold = threshold
        self.model_instance = None
        self.mean_vec = mean_vec

    def model(self):
        """Initialize and return the ViT model."""
        if self.model_instance is None:
            wt = torchvision.models.ViT_H_14_Weights.DEFAULT
            self.model_instance = vit_h_14(weights=wt)
            self.model_instance.heads = nn.Sequential(*list(self.model_instance.heads.children())[:-1])
            self.model_instance = self.model_instance.to(self.device)
        return self.model_instance

    def process_image(self, cv2_img):
        """
        Process a cv2 image for the model.

        Args:
            cv2_img: OpenCV image (BGR format)

        Returns:
            Processed tensor
        """
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_img = Image.fromarray(rgb_img)

        # A set of transformations to prepare the image in tensor format
        transformations = tr.Compose([
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            tr.Resize((518, 518))
        ])

        # preparing the image
        img_tensor = transformations(pil_img).float()

        if self.vector == 'image':
            img_tensor = img_tensor.flatten()

        img_tensor = img_tensor.unsqueeze_(0)

        if self.vector == 'feature':
            img_tensor = img_tensor.to(self.device)

        return img_tensor

    def get_embeddings(self, ref_images, test_images):
        """
        Get embeddings for reference and test images.

        Args:
            ref_images: List of cv2 reference images
            test_images: List of cv2 test images

        Returns:
            Reference embedding, list of test embeddings
        """
        model = self.model()

        # Process test images
        emb_test = []
        for img in test_images:
            processed_img = self.process_image(img)
            if self.vector == 'feature':
                emb = model(processed_img).detach().cpu()
                emb_test.append(emb)
            else:  # 'image'
                emb_test.append(processed_img)

        # This checks if a reference vector is loaded, if so the process of getting
        # reference embeddings can be skipped for efficiency
        if len(self.mean_vec) > 0:
            emb_ref = torch.tensor(self.mean_vec)

        # Process reference images if necessary
        else:
            if self.vector == 'feature':
                # Standard method of getting reference embedding vector
                emb_ref_list = []
                for img in ref_images:
                    processed_img = self.process_image(img)
                    emb = model(processed_img).detach().cpu()
                    emb_ref_list.append(emb)

                # Average the reference embeddings
                emb_ref = torch.mean(torch.stack(emb_ref_list), dim=0)
            
            else:  # 'image'
                emb_ref_list = []
                for img in ref_images:
                    processed_img = self.process_image(img)
                    emb_ref_list.append(processed_img)

                # Average the reference images
                emb_ref = torch.mean(torch.stack(emb_ref_list), dim=0)

        return emb_ref, emb_test

    def find_outliers(self, ref_images, test_images):
        """
        Find outliers in test images compared to reference images.

        Args:
            ref_images: List of cv2 reference images
            test_images: List of cv2 test images

        Returns:
            mask: Boolean array where True indicates an outlier
            scores: Similarity scores for each test image
        """
        emb_ref, emb_test = self.get_embeddings(ref_images, test_images)

        scores = []
        mask = []

        for i in range(len(emb_test)):
            score = torch.nn.functional.cosine_similarity(emb_ref, emb_test[i])
            score_value = score.item()
            scores.append(round(score_value, 4))
            # True if it's an outlier (below threshold)
            mask.append(score_value <= self.threshold)

        return np.array(mask), scores, emb_ref

    def filter_outliers(self, ref_images, test_images):
        """
        Filter out outliers from test images.

        Args:
            ref_images: List of cv2 reference images
            test_images: List of cv2 test images

        Returns:
            filtered_images: List of non-outlier test images
            outlier_mask: Boolean array where True indicates an outlier
            scores: Similarity scores for each test image
        """
        outlier_mask, scores, mean = self.find_outliers(ref_images, test_images)

        # Filter out outliers (keep only non-outliers)
        filtered_images = [img for i, img in enumerate(test_images) if not outlier_mask[i]]

        return filtered_images, outlier_mask, scores, mean

def detect_outliers(ref_imgs, imgs, mean_vec=[]):
    """
    Detects outliers in a set of test images, can use a reference vector

    Args:
        ref_images: List of cv2 reference images
        images: List of cv2 test images
        mean_vec: optional pre-computed reference vector

    Returns:
        filtered_images: List of non-outlier test images
        mean: the reference vector used (if a new reference vector should be saved)
    """
    
    similarity = CosineSimilarity(vector='feature', threshold=0.8, mean_vec=mean_vec)

    # Get outlier mask and scores
    outlier_mask, scores = similarity.find_outliers(ref_imgs, imgs)

    # Filter out outliers
    filtered_images, _, _, mean = similarity.filter_outliers(ref_imgs, imgs)

    return filtered_images, mean
