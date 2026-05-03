"""
Training loop and utilities for the multimodal fake news detection model
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, PolynomialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeNewsTrainer:
    """
    Trainer class for multimodal fake news detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config,
        device: str = 'cuda',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        self.model.to(device)
        
        # Loss function (with class weights for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Scheduler
        self.scheduler = self._setup_scheduler()
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components"""
        param_groups = [
            # Text encoder with lower learning rate
            {
                'params': self.model.encoders.text_encoder.parameters(),
                'lr': self.config.training.learning_rate * 0.5,
                'weight_decay': self.config.training.weight_decay,
            },
            # Image encoder with medium learning rate
            {
                'params': self.model.encoders.image_encoder.parameters(),
                'lr': self.config.training.learning_rate * 0.75,
                'weight_decay': self.config.training.weight_decay,
            },
            # Fusion and classifier with higher learning rate
            {
                'params': list(self.model.fusion.parameters()) + list(self.model.classifier.parameters()),
                'lr': self.config.training.learning_rate,
                'weight_decay': self.config.training.weight_decay,
            },
        ]
        
        return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.training.learning_rate_scheduler == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7,
            )
        elif self.config.training.learning_rate_scheduler == 'linear':
            return LinearLR(self.optimizer, total_iters=self.config.training.num_epochs)
        elif self.config.training.learning_rate_scheduler == 'polynomial':
            return PolynomialLR(self.optimizer, total_iters=self.config.training.num_epochs, power=1.0)
        else:
            raise ValueError(f"Unknown scheduler: {self.config.training.learning_rate_scheduler}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, images)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            # Predictions
            preds = torch.argmax(outputs['probs'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Optimizer step
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Metrics
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, images)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs['probs'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    @staticmethod
    def _compute_metrics(labels: list, preds: list) -> Dict[str, float]:
        """Compute evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        labels = np.array(labels)
        preds = np.array(preds)
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        }
    
    def train(self, num_epochs: Optional[int] = None):
        """Complete training loop"""
        num_epochs = num_epochs or self.config.training.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('train/f1', train_metrics['f1'], epoch)
            
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(f"best_f1_epoch_{epoch}.pt")
                logger.info(f"✓ Saved best model with F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= 5:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = Path(self.config.MODEL_DIR) / filename
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = Path(self.config.MODEL_DIR) / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
    
    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, images)
                
                preds = torch.argmax(outputs['probs'], dim=1)
                confidences = torch.max(outputs['probs'], dim=1)[0]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['mean_confidence'] = float(np.mean(all_confidences))
        
        return metrics
