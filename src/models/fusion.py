"""
Multimodal Fusion Layers - Unique architectures for combining modalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class AdaptiveFusion(nn.Module):
    """
    Adaptive Fusion Layer with learnable weights and gating mechanism
    
    This is more unique than simple concatenation - it learns the importance
    of each modality dynamically based on input.
    """
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Modality-specific transformations
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.image_transform = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Gating mechanism (learns importance of each modality)
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.image_gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Fusion blocks
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: [batch_size, text_dim]
            image_emb: [batch_size, image_dim]
        
        Returns:
            Dict with fused embeddings and attention weights
        """
        # Transform modalities
        text_transformed = self.text_transform(text_emb)  # [batch_size, hidden_dim]
        image_transformed = self.image_transform(image_emb)  # [batch_size, hidden_dim]
        
        # Concatenate for gating
        combined = torch.cat([text_emb, image_emb], dim=1)  # [batch_size, text_dim + image_dim]
        
        # Calculate gates (learns which modality is more important)
        text_gate = self.text_gate(combined)  # [batch_size, 1]
        image_gate = self.image_gate(combined)  # [batch_size, 1]
        
        # Normalize gates
        gate_sum = text_gate + image_gate
        text_gate = text_gate / (gate_sum + 1e-8)
        image_gate = image_gate / (gate_sum + 1e-8)
        
        # Apply gated fusion
        gated_text = text_transformed * text_gate
        gated_image = image_transformed * image_gate
        
        # Concatenate gated representations
        fused = torch.cat([gated_text, gated_image], dim=1)  # [batch_size, 2 * hidden_dim]
        
        # Apply fusion blocks
        for fusion_block in self.fusion_blocks:
            fused = fusion_block(fused)
        
        # Cross-modal attention
        text_q = text_transformed.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        image_kv = image_transformed.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attended, attention_weights = self.cross_attention(
            query=text_q,
            key=image_kv,
            value=image_kv
        )
        
        fused = fused + attended.squeeze(1)
        
        return {
            'fused': fused,
            'text_gate': text_gate,
            'image_gate': image_gate,
            'attention_weights': attention_weights,
        }


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion - Bi-directional attention between modalities
    """
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Cross-attention (text attends to image)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention (image attends to text)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: [batch_size, text_dim]
            image_emb: [batch_size, image_dim]
        
        Returns:
            Dict with fused embeddings
        """
        # Project to common dimension
        text_h = self.text_proj(text_emb).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        image_h = self.image_proj(image_emb).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention
        text_attended, text_attn = self.text_to_image_attn(
            query=text_h,
            key=image_h,
            value=image_h
        )
        
        image_attended, image_attn = self.image_to_text_attn(
            query=image_h,
            key=text_h,
            value=text_h
        )
        
        # Fuse attended representations
        fused = torch.cat([text_attended, image_attended], dim=-1)  # [batch_size, 1, 2*hidden_dim]
        fused = fused.squeeze(1)  # [batch_size, 2*hidden_dim]
        
        fused = self.fusion_norm(self.fusion_mlp(fused))
        
        return {
            'fused': fused,
            'text_attention': text_attn,
            'image_attention': image_attn,
        }


class TensorFusionNetwork(nn.Module):
    """
    Tensor Fusion Network - Advanced fusion using outer product
    Based on the paper: Tensor Fusion Network for Multimodal Learning
    """
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Bilinear fusion
        self.fusion_weight = nn.Parameter(
            torch.randn(text_dim + 1, image_dim + 1, hidden_dim) / 100
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: [batch_size, text_dim]
            image_emb: [batch_size, image_dim]
        
        Returns:
            Dict with fused embeddings
        """
        batch_size = text_emb.shape[0]
        
        # Add bias terms
        text_biased = torch.cat([text_emb, torch.ones(batch_size, 1, device=text_emb.device)], dim=1)
        image_biased = torch.cat([image_emb, torch.ones(batch_size, 1, device=image_emb.device)], dim=1)
        
        # Tensor fusion
        fused = torch.zeros(batch_size, self.hidden_dim, device=text_emb.device)
        
        for i in range(batch_size):
            # Compute outer product: text * image^T
            outer = torch.outer(text_biased[i], image_biased[i])  # [(text_dim+1), (image_dim+1)]
            # Reshape and multiply with weights
            outer_flat = outer.reshape(-1)  # [(text_dim+1)*(image_dim+1)]
            fused_weights = self.fusion_weight.reshape(-1, self.hidden_dim)  # [(text_dim+1)*(image_dim+1), hidden_dim]
            fused[i] = torch.matmul(outer_flat, fused_weights)
        
        fused = self.output_proj(fused)
        
        return {'fused': fused}


class MultimodalFusion(nn.Module):
    """
    Main fusion layer that can switch between different fusion strategies
    """
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, fusion_type: str = "adaptive", **kwargs):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "adaptive":
            self.fusion = AdaptiveFusion(text_dim, image_dim, hidden_dim, **kwargs)
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(text_dim, image_dim, hidden_dim, **kwargs)
        elif fusion_type == "tensor":
            self.fusion = TensorFusionNetwork(text_dim, image_dim, hidden_dim, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.fusion(text_emb, image_emb)
