"""
Text and Image Encoders for multimodal fusion
"""
import torch
import torch.nn as nn
from transformers import AutoModel
import timm
from typing import Dict, Tuple


class TextEncoder(nn.Module):
    """
    Text Encoder using RoBERTa with pooling and projection
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained model
        self.backbone = AutoModel.from_pretrained(config.text.model_name)
        self.backbone.config.output_hidden_states = True
        
        # Freeze early layers (optional but recommended)
        self._freeze_early_layers()
        
        # Pooling layers
        self.attention_pooling = nn.Linear(768, 1)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(768, config.text.hidden_size),
            nn.LayerNorm(config.text.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.text.dropout),
        )
        
    def _freeze_early_layers(self, num_layers_to_freeze: int = 6):
        """Freeze early transformer layers to reduce computation"""
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        
        Returns:
            embeddings: Projected embeddings [batch_size, hidden_size]
            hidden_states: Hidden states for attention visualization [batch_size, seq_length, 768]
        """
        # Get model outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get hidden states for all layers
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, 768]
        
        # Attention-based pooling
        attention_weights = torch.softmax(
            self.attention_pooling(hidden_states),  # [batch_size, seq_length, 1]
            dim=1
        )
        pooled = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, 768]
        
        # Project
        embeddings = self.projection(pooled)  # [batch_size, hidden_size]
        
        return embeddings, hidden_states


class ImageEncoder(nn.Module):
    """
    Image Encoder using EfficientNet with feature pyramid
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained model
        self.backbone = timm.create_model(
            config.image.model_name,
            pretrained=True,
            features_only=False,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.image.image_size, config.image.image_size)
            dummy_output = self.backbone(dummy_input)
            feature_dim = dummy_output.shape[-1] if len(dummy_output.shape) > 1 else dummy_output.shape[1]
        
        # Feature pyramid and refinement
        self.feature_refine = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if len(dummy_output.shape) > 2 else nn.Identity(),
            nn.Flatten() if len(dummy_output.shape) > 2 else nn.Identity(),
        )
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, config.image.hidden_size),
            nn.LayerNorm(config.image.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.image.dropout),
        )
        
        # Freeze early layers (optional)
        self._freeze_early_layers()
    
    def _freeze_early_layers(self, freeze_ratio: float = 0.5):
        """Freeze early layers to reduce computation"""
        params = list(self.backbone.parameters())
        num_to_freeze = int(len(params) * freeze_ratio)
        
        for i, param in enumerate(params):
            if i < num_to_freeze:
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Image tensor [batch_size, 3, height, width]
        
        Returns:
            embeddings: Projected embeddings [batch_size, hidden_size]
            features: Raw features for attention visualization
        """
        # Extract features
        features = self.backbone(images)  # Could be different shapes based on model
        
        # Refine features
        refined = self.feature_refine(features)
        if len(refined.shape) == 1:
            refined = refined.unsqueeze(0)
        
        # Project
        embeddings = self.projection(refined)  # [batch_size, hidden_size]
        
        return embeddings, features


class MultimodalEncoder(nn.Module):
    """
    Combined text and image encoders with shared interface
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            images: Images [batch_size, 3, height, width]
        
        Returns:
            Dictionary with:
                text_embeddings: [batch_size, text_hidden_size]
                image_embeddings: [batch_size, image_hidden_size]
                text_hidden_states: For visualization
                image_features: For visualization
        """
        text_embeddings, text_hidden_states = self.text_encoder(input_ids, attention_mask)
        image_embeddings, image_features = self.image_encoder(images)
        
        return {
            'text_embeddings': text_embeddings,
            'image_embeddings': image_embeddings,
            'text_hidden_states': text_hidden_states,
            'image_features': image_features,
        }
