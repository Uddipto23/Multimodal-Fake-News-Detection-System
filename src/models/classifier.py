"""
Classification head for binary fake news detection
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class FakeNewsClassifier(nn.Module):
    """
    Binary classifier with confidence estimation and uncertainty quantification
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification head with residual connections
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer
        self.logits = nn.Linear(current_dim, num_classes)
        
        # Uncertainty head (epistemic uncertainty)
        self.uncertainty = nn.Sequential(
            nn.Linear(current_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures positive uncertainty
        )
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            embeddings: Fused embeddings from fusion layer [batch_size, input_dim]
        
        Returns:
            Dict with:
                logits: Raw predictions [batch_size, num_classes]
                probs: Softmax probabilities [batch_size, num_classes]
                confidence: Model confidence [batch_size, 1]
                uncertainty: Uncertainty estimate [batch_size, 1]
        """
        # Extract features
        features = self.feature_extractor(embeddings)
        
        # Get logits
        logits = self.logits(features)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get confidence (max probability)
        confidence, _ = torch.max(probs, dim=1, keepdim=True)
        
        # Get uncertainty
        uncertainty = self.uncertainty(features)
        
        return {
            'logits': logits,
            'probs': probs,
            'confidence': confidence,
            'uncertainty': uncertainty,
        }


class MultimodalFakeNewsModel(nn.Module):
    """
    Complete multimodal fake news detection model
    
    Architecture:
    Text Input -> Text Encoder -> |
                                  | -> Fusion Layer -> Classifier -> Output
    Image Input -> Image Encoder -> |
    """
    
    def __init__(self, encoders, fusion, classifier):
        super().__init__()
        
        self.encoders = encoders
        self.fusion = fusion
        self.classifier = classifier
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            images: Images [batch_size, 3, height, width]
        
        Returns:
            Dict with model outputs
        """
        # Encode modalities
        encoder_outputs = self.encoders(input_ids, attention_mask, images)
        
        # Fuse
        fusion_outputs = self.fusion(
            encoder_outputs['text_embeddings'],
            encoder_outputs['image_embeddings']
        )
        
        # Classify
        classifier_outputs = self.classifier(fusion_outputs['fused'])
        
        # Combine outputs
        return {
            **encoder_outputs,
            **fusion_outputs,
            **classifier_outputs,
        }
    
    def get_predictions(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get class predictions from logits
        
        Args:
            logits: Raw model outputs [batch_size, num_classes]
        
        Returns:
            predictions: Predicted classes [batch_size]
            confidences: Confidence scores [batch_size]
        """
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        return predictions, confidences


class MonteCarloDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation
    Use during inference with dropout enabled to get stochastic predictions
    """
    
    def __init__(self, model: nn.Module, num_iterations: int = 10):
        super().__init__()
        self.model = model
        self.num_iterations = num_iterations
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate multiple predictions with MC Dropout
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            images: Images [batch_size, 3, height, width]
        
        Returns:
            Dict with mean and std of predictions
        """
        self.model.train()  # Keep dropout enabled
        
        predictions = []
        confidences = []
        
        for _ in range(self.num_iterations):
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, images)
                predictions.append(outputs['probs'])
                confidences.append(outputs['confidence'])
        
        # Stack and compute statistics
        predictions = torch.stack(predictions)  # [num_iterations, batch_size, num_classes]
        confidences = torch.stack(confidences)  # [num_iterations, batch_size, 1]
        
        mean_probs = torch.mean(predictions, dim=0)
        std_probs = torch.std(predictions, dim=0)
        mean_confidence = torch.mean(confidences, dim=0)
        std_confidence = torch.std(confidences, dim=0)
        
        self.model.eval()
        
        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'epistemic_uncertainty': std_probs.mean(dim=1, keepdim=True),
        }
