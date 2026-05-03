"""
Configuration file for Multimodal Fake News Detection System
"""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class TextConfig:
    """Text model configuration"""
    model_name: str = "roberta-base"  # Changed from BERT for better performance
    max_length: int = 256
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 2e-5
    
@dataclass
class ImageConfig:
    """Image model configuration"""
    model_name: str = "efficientnet_b3"  # Lightweight yet powerful
    image_size: int = 384
    hidden_size: int = 1536
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-4
    
@dataclass
class FusionConfig:
    """Fusion layer configuration"""
    fusion_type: str = "adaptive"  # Options: concat, adaptive, cross_attention
    hidden_size: int = 512
    num_fusion_layers: int = 3
    dropout: float = 0.2
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 16
    num_epochs: int = 15
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    learning_rate_scheduler: str = "cosine"  # Options: linear, cosine, polynomial
    device: str = "cuda"
    seed: int = 42
    
@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "./data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    
@dataclass
class ExplainabilityConfig:
    """Explainability configuration"""
    use_lime: bool = True
    use_shap: bool = True
    use_attention_visualization: bool = True
    num_samples_lime: int = 1000
    
class Config:
    """Master configuration class"""
    text: TextConfig = TextConfig()
    image: ImageConfig = ImageConfig()
    fusion: FusionConfig = FusionConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    
    # Paths
    MODEL_DIR = "./models"
    LOG_DIR = "./logs"
    OUTPUT_DIR = "./outputs"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)

# Initialize config
config = Config()
