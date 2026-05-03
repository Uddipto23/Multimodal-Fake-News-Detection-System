"""
Google Colab training notebook for Multimodal Fake News Detection System
Run this in Google Colab for easy training and experimentation
"""

# Install dependencies
# !pip install -q torch torchvision transformers opencv-python albumentations timm
# !pip install -q lime shap tensorboard scikit-learn

# Clone repository (if using from GitHub)
# !git clone https://github.com/Uddipto23/Multimodal-Fake-News-Detection-System.git
# %cd Multimodal-Fake-News-Detection-System

# Imports
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Check GPU availability
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# ============================================================================
# SECTION 1: Setup and Data Preparation
# ============================================================================

from config import config
from src.data.dataset import FakeNewsDataLoader, MultimodalFakeNewsDataset
from src.models.encoders import MultimodalEncoder
from src.models.fusion import MultimodalFusion
from src.models.classifier import FakeNewsClassifier, MultimodalFakeNewsModel
from src.training.trainer import FakeNewsTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# SECTION 2: Generate Sample Dataset
# ============================================================================

print("\n" + "="*80)
print("CREATING SAMPLE DATASET")
print("="*80)

# Create data directories
data_dir = Path('./data')
for split in ['train', 'val', 'test']:
    (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)

# Generate synthetic dataset for demonstration
def create_sample_dataset(num_train=100, num_val=20, num_test=20):
    """Create synthetic data for demonstration"""
    import cv2
    
    texts_samples = [
        "Breaking: New medical breakthrough discovered in laboratory studies.",
        "Expert reveals shocking truth about common household items.",
        "Scientists warn of unprecedented climate catastrophe.",
        "Local business thrives with innovative business model.",
        "Celebrity spotted with mystery person - details inside.",
        "Government announces new policy affecting millions.",
        "SHOCKING: This one weird trick doctors hate!",
        "Fake headlines generator creates misleading news.",
        "This is completely false and untrue information.",
    ]
    
    for split, num_samples in [('train', num_train), ('val', num_val), ('test', num_test)]:
        split_dir = data_dir / split
        samples = []
        
        for i in range(num_samples):
            # Generate fake image (random noise)
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image_path = split_dir / 'images' / f'{i}.jpg'
            cv2.imwrite(str(image_path), image)
            
            # Generate sample metadata
            text = texts_samples[i % len(texts_samples)]
            label = i % 2  # Alternating fake/real
            
            samples.append({
                'id': str(i),
                'text': text,
                'image_id': str(i),
                'label': label,
            })
        
        # Save JSONL file
        jsonl_path = split_dir / 'articles.jsonl'
        with open(jsonl_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Created {num_samples} samples for {split} split")

create_sample_dataset(num_train=100, num_val=20, num_test=20)

# ============================================================================
# SECTION 3: Create Data Loaders
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_loader, val_loader, test_loader = FakeNewsDataLoader.create_dataloaders(
    data_dir=str(data_dir),
    text_config=config.text,
    image_config=config.image,
    batch_size=config.training.batch_size,
    num_workers=0,  # Use 0 for Colab
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Test dataloader
batch = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  input_ids: {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
print(f"  image: {batch['image'].shape}")
print(f"  label: {batch['label'].shape}")

# ============================================================================
# SECTION 4: Initialize Model
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING MODEL")
print("="*80)

# Create encoders
encoders = MultimodalEncoder(config)
print(f"✓ Text Encoder: {config.text.model_name}")
print(f"✓ Image Encoder: {config.image.model_name}")

# Create fusion layer
fusion = MultimodalFusion(
    text_dim=config.text.hidden_size,
    image_dim=config.image.hidden_size,
    hidden_dim=config.fusion.hidden_size,
    fusion_type=config.fusion.fusion_type,
    num_layers=config.fusion.num_fusion_layers,
    dropout=config.fusion.dropout,
)
print(f"✓ Fusion Layer: {config.fusion.fusion_type}")

# Create classifier
classifier = FakeNewsClassifier(
    input_dim=config.fusion.hidden_size,
    num_classes=2,
    hidden_dim=256,
    dropout=config.fusion.dropout,
)
print(f"✓ Classifier head")

# Create complete model
model = MultimodalFakeNewsModel(encoders, fusion, classifier)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# ============================================================================
# SECTION 5: Training
# ============================================================================

print("\n" + "="*80)
print("TRAINING")
print("="*80)

# Create trainer
trainer = FakeNewsTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    config=config,
    device=device,
)

# Train
trainer.train(num_epochs=5)  # Reduced for demo

# Save final model
trainer.save_checkpoint("final_model.pt")

# ============================================================================
# SECTION 6: Evaluation
# ============================================================================

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

test_metrics = trainer.evaluate_test()
print(f"\nTest Set Metrics:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1-Score: {test_metrics['f1']:.4f}")

# ============================================================================
# SECTION 7: Explainability Demo
# ============================================================================

print("\n" + "="*80)
print("EXPLAINABILITY DEMO")
print("="*80)

from src.explainability.explainer import MultimodalExplainer

explainer = MultimodalExplainer(model, device)

# Get a sample from test set
batch = next(iter(test_loader))
input_ids = batch['input_ids'][:1].to(device)
attention_mask = batch['attention_mask'][:1].to(device)
images = batch['image'][:1].to(device)

# Get explanations
explanations = explainer.explain_prediction(input_ids, attention_mask, images)
print("✓ Generated explanations")

# Get predictions
with torch.no_grad():
    outputs = model(input_ids, attention_mask, images)
    probs = outputs['probs'][0].cpu().numpy()
    prediction = int(np.argmax(probs))

print(f"\nPrediction: {'FAKE' if prediction == 1 else 'REAL'}")
print(f"Real Probability: {probs[0]:.4f}")
print(f"Fake Probability: {probs[1]:.4f}")

# ============================================================================
# SECTION 8: Visualizations
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot image and saliency
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original image
image_np = images[0].cpu().numpy().transpose(1, 2, 0)
axes[0].imshow(image_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Saliency map
saliency = explanations['image_gradient'][0]
axes[1].imshow(saliency, cmap='hot')
axes[1].set_title('Image Saliency Map')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('./outputs/saliency_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved saliency visualization")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModels saved in: {config.MODEL_DIR}")
print(f"Logs saved in: {config.LOG_DIR}")
print(f"Outputs saved in: {config.OUTPUT_DIR}")
