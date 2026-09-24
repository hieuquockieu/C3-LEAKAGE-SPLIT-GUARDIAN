"""
embedding_detector.py
Deep Visual Embedding Leakage Detector.

Extracts dense L2-normalized feature vectors for image samples and computes
pairwise cosine similarities to identify semantic and viewpoint duplicates.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

class LightweightFeatureExtractor(nn.Module):
    """
    Calibrated visual feature extractor that captures spatial color distributions,
    structural gradients, and convolutional features into a 256-dimensional L2-normalized vector.
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Convolutional texture branch
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # 64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)) # 32 * 4 = 128
        )
        self.proj = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 3, 128, 128)
        B, C, H, W = x.shape

        # 1. Spatial color moments: 4x4 spatial grid
        grid_h, grid_w = H // 4, W // 4
        # Reshape to (B, 3, 4, grid_h, 4, grid_w)
        blocks = x.view(B, C, 4, grid_h, 4, grid_w).permute(0, 2, 4, 1, 3, 5).contiguous()
        # Mean across grid cells (B, 4, 4, 3) -> (B, 48)
        spatial_means = blocks.mean(dim=(4, 5)).view(B, -1)
        spatial_stds = blocks.std(dim=(4, 5)).view(B, -1)
        color_feats = torch.cat([spatial_means, spatial_stds], dim=1) # (B, 96)

        # 2. Gradient features (horizontal and vertical differences)
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean(dim=(2, 3)) # (B, 3)
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean(dim=(2, 3)) # (B, 3)
        grad_feats = torch.cat([diff_h, diff_w], dim=1) # (B, 6)
        # Pad grad_feats to 32
        grad_padded = torch.nn.functional.pad(grad_feats, (0, 26))

        # 3. Conv features (128)
        conv_feats = self.proj(self.conv(x).flatten(1)) # (B, 128)

        # Combine all to 256
        combined = torch.cat([color_feats, grad_padded, conv_feats], dim=1) # 96 + 32 + 128 = 256
        # L2-normalization for cosine similarity
        norm = torch.norm(combined, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return combined / norm

class EmbeddingDetector:
    def __init__(self, sim_threshold: float = 0.84, device: str = "cpu"):
        self.sim_threshold = sim_threshold
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Initialize deterministic feature extractor
        torch.manual_seed(42)
        self.model = LightweightFeatureExtractor(embedding_dim=256).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        return emb.squeeze(0).cpu().numpy()

    def compute_all_embeddings(
        self,
        samples: List[Dict],
        images_lookup: Dict[str, Image.Image]
    ) -> Dict[str, np.ndarray]:
        sample_ids = [s["sample_id"] for s in samples]
        raw_list = []
        for sid in sample_ids:
            img = images_lookup[sid]
            raw_list.append(self.extract_embedding(img))
        
        raw_matrix = np.array(raw_list)
        # Zero-center features across dataset to calibrate cosine similarity metric
        centered = raw_matrix - np.mean(raw_matrix, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        normalized = centered / norms

        embeddings = {sid: normalized[i] for i, sid in enumerate(sample_ids)}
        return embeddings

    def detect_pairwise_leakage(
        self,
        samples: List[Dict],
        embeddings: Dict[str, np.ndarray]
    ) -> List[Tuple[str, str, float]]:
        """
        Calculates all-pairs cosine similarity matrix and returns pairs exceeding threshold.
        """
        sample_ids = [s["sample_id"] for s in samples]
        n = len(sample_ids)
        emb_matrix = np.array([embeddings[sid] for sid in sample_ids]) # shape (N, D)

        # Dot product of L2-normalized embeddings equals cosine similarity
        sim_matrix = np.dot(emb_matrix, emb_matrix.T)

        predicted_leak_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= self.sim_threshold:
                    predicted_leak_pairs.append((sample_ids[i], sample_ids[j], sim))

        return predicted_leak_pairs
