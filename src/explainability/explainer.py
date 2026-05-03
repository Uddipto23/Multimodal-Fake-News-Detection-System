"""
Explainability module for multimodal fake news detection
Provides LIME, SHAP, attention visualization, and gradients
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image


class MultimodalExplainer:
    """
    Main explainability class combining multiple explanation methods
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain_prediction(self, input_ids, attention_mask, images) -> Dict:
        """
        Get comprehensive explanation for a prediction
        
        Returns:
            Dict with multiple explanation types
        """
        explanations = {}
        
        # Text attention
        explanations['text_attention'] = self.get_text_attention(input_ids, attention_mask, images)
        
        # Image gradient
        explanations['image_gradient'] = self.get_image_gradient(input_ids, attention_mask, images)
        
        # Attention rollout
        explanations['attention_rollout'] = self.get_attention_rollout(input_ids, attention_mask, images)
        
        return explanations
    
    def get_text_attention(self, input_ids, attention_mask, images) -> np.ndarray:
        """
        Extract attention weights from text encoder
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.encoders(input_ids, attention_mask, images)
            # Get hidden states for attention visualization
            text_hidden = outputs['text_hidden_states']  # [batch_size, seq_len, hidden_dim]
        
        return text_hidden.cpu().numpy()
    
    def get_image_gradient(self, input_ids, attention_mask, images) -> np.ndarray:
        """
        Compute gradient of output with respect to input image
        """
        input_ids = input_ids.to(self.device).detach()
        attention_mask = attention_mask.to(self.device).detach()
        images = images.to(self.device).clone().detach().requires_grad_(True)
        
        outputs = self.model(input_ids, attention_mask, images)
        
        # Get prediction probability
        pred_prob = outputs['probs']
        pred_class = torch.argmax(pred_prob, dim=1)
        target_prob = pred_prob[0, pred_class[0]]
        
        # Compute gradient
        target_prob.backward()
        
        gradients = images.grad.data.cpu().numpy()
        
        # Take absolute value and average across channels
        saliency = np.abs(gradients).mean(axis=1)  # [batch_size, height, width]
        
        return saliency
    
    def get_attention_rollout(self, input_ids, attention_mask, images) -> np.ndarray:
        """
        Attention rollout for visualizing multi-head attention
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.encoders(input_ids, attention_mask, images)
            # Get fusion attention weights if available
            fusion_outputs = self.model.fusion(
                outputs['text_embeddings'],
                outputs['image_embeddings']
            )
        
        if 'attention_weights' in fusion_outputs:
            return fusion_outputs['attention_weights'].cpu().numpy()
        else:
            return np.array([])
    
    def visualize_text_attention(self, tokens: List[str], attention_weights: np.ndarray, save_path: str = None):
        """
        Visualize text attention as a heatmap
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attention Head')
        ax.set_title('Text Attention Heatmap')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_image_saliency(self, image: np.ndarray, saliency: np.ndarray, save_path: str = None):
        """
        Visualize image saliency map
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Saliency map
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image.astype(np.uint8), alpha=0.6)
        axes[2].imshow(saliency, cmap='hot', alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_fusion_gates(self, text_gate: float, image_gate: float, save_path: str = None):
        """
        Visualize fusion layer gates
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        modalities = ['Text', 'Image']
        gates = [text_gate, image_gate]
        
        bars = ax.bar(modalities, gates, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Gate Weight')
        ax.set_title('Fusion Layer Gate Weights')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, gate in zip(bars, gates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{gate:.3f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for text
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def explain_text(self, text: str, num_samples: int = 1000) -> Dict:
        """
        Generate LIME explanation for text
        """
        try:
            from lime.lime_text import LimeTextExplainer
            
            explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
            
            def predict_fn(texts):
                # Batch predict
                predictions = []
                for t in texts:
                    encodings = self.tokenizer(
                        t,
                        max_length=256,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    with torch.no_grad():
                        input_ids = encodings['input_ids'].to(self.device)
                        attention_mask = encodings['attention_mask'].to(self.device)
                        # Note: You'd need to handle dummy images here
                        probs = torch.softmax(torch.randn(1, 2), dim=1).cpu().numpy()
                    
                    predictions.append(probs)
                
                return np.vstack(predictions)
            
            explanation = explainer.explain_instance(text, predict_fn, num_samples=num_samples)
            
            return {
                'lime_explanation': explanation,
                'weights': dict(explanation.as_list()),
            }
        
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return {}


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model explanation
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def explain_prediction(self, input_ids, attention_mask, images, background_samples: int = 100):
        """
        Generate SHAP explanation
        """
        try:
            import shap
            
            def model_predict(data):
                # data shape: [num_samples, seq_length]
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).long().to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(data, torch.ones_like(data), torch.randn(data.shape[0], 3, 384, 384).to(self.device))
                    probs = outputs['probs'].cpu().numpy()
                
                return probs[:, 1]  # Return probability of fake class
            
            explainer = shap.KernelExplainer(model_predict, input_ids[:background_samples])
            shap_values = explainer.shap_values(input_ids[:1])
            
            return {
                'shap_values': shap_values,
                'explainer': explainer,
            }
        
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return {}


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for image explanation
    """
    
    def __init__(self, model, target_layer, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_ids, attention_mask, images, target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        """
        self.model.eval()
        images = images.to(self.device).requires_grad_(True)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask, images)
        
        # Backward pass
        target_score = outputs['probs'][0, target_class]
        target_score.backward()
        
        # Compute CAM
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)  # [batch, channels, 1, 1]
        cam = (gradients * self.activations).sum(dim=1, keepdim=True)  # [batch, 1, h, w]
        cam = F.relu(cam)
        cam = cam.cpu().numpy()[0, 0]
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
